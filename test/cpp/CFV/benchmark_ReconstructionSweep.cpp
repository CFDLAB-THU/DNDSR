// Fixed-input scans of actual prepared CFV publications. Timing excludes
// construction and placement; the locality leg uses the normal Geom transition.
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "CFV/Detail/ReconstructionDeviceKernels.hpp"
#include "Geom/IO/NativeMeshCGNS.hpp"
#include "Geom/Transition/LocalCellOrdering.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <numeric>

namespace C = DNDS::CFV;
namespace D = DNDS::CFV::Detail;
namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;
using J = nlohmann::ordered_json;
static void Check(cudaError_t result)
{
    DNDS_check_throw_info(result == cudaSuccess, cudaGetErrorString(result));
}
static void Run(int argc, char **argv)
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    DNDS_check_throw_info(mpi.size == 1 && argc == 6, "usage: sweep_benchmark periodic-mesh catalog.json precomputed|raw-full|raw-triangle|quantized subpartitions repetitions (np1)");
    const std::string representation = argv[3];
    const int partitions = std::stoi(argv[4]), repetitions = std::stoi(argv[5]);
    DNDS_check_throw_info(partitions >= 0 && repetitions > 0, "invalid sweep benchmark controls");
    J catalog;
    std::ifstream(argv[2]) >> catalog;
    G::NativePeriodicImportOptions periodic;
    constexpr double length = 6.2831853071795864769;
    for (int d = 0; d < 3; ++d)
    {
        periodic.frames.translation[d + 1].map().setZero();
        periodic.frames.translation[d + 1].map()[d] = length;
    }
    const double setupStart = MPI_Wtime();
    auto mesh = G::UnstructuredMesh::FromNative(mpi, G::ReadNativeCGNS(mpi, argv[1], 3, G::FBC_Name_2_ID_Default, 0, {1, true}, periodic));
    mesh->PrepareNativeTopology();
    const double imported = MPI_Wtime();
    if (partitions)
        mesh->ReorderLocalCells(G::LocalCellOrderingSettings{partitions, G::CellCoreSplit::None, false});
    const double reordered = MPI_Wtime();
    C::ReconstructionSettings settings;
    settings.maxOrder = 3;
    settings.intOrder = 5;
    settings.cacheDiffBase = false;
    settings.functional.dirWeightScheme = C::ReconstructionFunctionalSettings::DirectionalWeight::HQM_OPT;
    settings.functional.geomWeightScheme = C::ReconstructionFunctionalSettings::GeometricWeight::HQM_SD;
    settings.functional.scaleType = C::ReconstructionFunctionalSettings::LengthScale::BaryDiff;
    if (representation != "precomputed" && representation != "quantized")
    {
        DNDS_check_throw_info(representation == "raw-full" || representation == "raw-triangle", "unknown sweep representation");
        settings.storage.representation = "raw";
        settings.storage.faceCoupling = true;
        settings.storage.inverseTriangle = representation == "raw-triangle";
    }
    C::FiniteVolumeMetricOptions metricOptions;
    metricOptions.integrationOrder = 5;
    metricOptions.collectivePreflight = false;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh, metricOptions);
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings, C::ReconstructionCapabilities::ForSweeps(false));
    const double prepared = MPI_Wtime();
    const auto device = R::Device::CUDA(0);
    R::EagerEngine engine;
    const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    const auto context = stream.Context();
    auto resident = plan.Place(device, context);
    double placed = MPI_Wtime();
    constexpr I components = 5;
    auto mean = metrics.BuildField(components), coefficients = plan.BuildCoefficients(components);
    {
        const auto state = mesh->CurrentNativeState();
        auto mw = mean.Owned().Write(), cw = coefficients.Owned().Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            const I id = state->OriginalEntityId(G::EntityKind::Cell, iCell);
            for (I j = 0; j < components; ++j)
                mw.View()(iCell, j) = 1 + .001 * ((id * 17 + j * 7) % 101);
            for (I j = 0; j < components * plan.NRecDOF(); ++j)
                cw.View()(iCell, j) = .01 * (1 + .001 * ((id * 13 + j * 7) % 101));
        }
    }
    metrics.BindHalo(mean).Begin().Wait();
    metrics.BindHalo(coefficients).Begin().Wait();
    auto dm = metrics.BuildField(R::RowLayout::Dense(0, components), device);
    auto du = resident.BuildCoefficients(components);
    for (auto pair : {std::make_pair(&dm, &mean), std::make_pair(&du, &coefficients)})
    {
        pair.first->Owned().CopyFrom(pair.second->Owned(), context).Wait();
        pair.first->Ghost().CopyFrom(pair.second->Ghost(), context).Wait();
    }
    const auto boundary = plan.BuildBoundaryValues(components);
    auto db = resident.PlaceBoundaryValues(boundary);
    auto workspace = resident.BuildWorkspace(components);
    resident.SubmitReconstructionIterTo(engine, stream, dm, du, workspace, db, C::ReconstructionRegion::All, C::ReconstructionDeviceKernel::FP64);
    engine.Synchronize();
    const auto reference = workspace.temporary.Owned().To(R::Device::Host());
    if (const auto *snapshot = std::getenv("DNDS_VR_SWEEP_SNAPSHOT"))
    {
        const auto state = mesh->CurrentNativeState();
        std::vector<I> rows(size_t(metrics.NumCell()));
        std::iota(rows.begin(), rows.end(), I(0));
        std::sort(rows.begin(), rows.end(), [&](I a, I b)
                  { return state->OriginalEntityId(G::EntityKind::Cell, a) < state->OriginalEntityId(G::EntityKind::Cell, b); });
        std::ofstream file(snapshot, std::ios::binary);
        const I shape[]{metrics.NumCell(), components * plan.NRecDOF()};
        file.write(reinterpret_cast<const char *>(shape), sizeof(shape));
        const auto read = reference.Read();
        for (const auto iCell : rows)
        {
            const auto id = state->OriginalEntityId(G::EntityKind::Cell, iCell);
            file.write(reinterpret_cast<const char *>(&id), sizeof(id));
            file.write(reinterpret_cast<const char *>(read.View()[iCell].data()), shape[1] * sizeof(double));
        }
        DNDS_check_throw_info(file.good(), "cannot write persistent-ID sweep snapshot");
    }
    cudaDeviceProp properties{};
    Check(cudaGetDeviceProperties(&properties, 0));

    if (representation == "quantized")
    {
        const double quantizedStart = MPI_Wtime();
        C::ReconstructionResidencyOptions options;
        options.matrixPrecision = C::ReconstructionMatrixPrecision::FP32;
        resident = plan.Place(device, context, options);
        placed += MPI_Wtime() - quantizedStart;
        db = resident.PlaceBoundaryValues(boundary);
        workspace = resident.BuildWorkspace(components);
    }
    // StorageReport describes retained host preparation when host payloads are
    // available. This receipt needs the selected device publication instead,
    // particularly after quantization; never label FP64 host bytes as FP32.
    size_t residentBytes = 0;
    for (const auto &buffer : resident.ResidentBuffers())
        residentBytes += buffer.bytes;
    std::cout << "VR_SETUP " << J{{"device", properties.name}, {"cells", metrics.NumCell()}, {"subpartitions", partitions}, {"representation", representation}, {"import_seconds", imported - setupStart}, {"reorder_seconds", reordered - imported}, {"prepare_seconds", prepared - reordered}, {"placement_seconds", placed - prepared}, {"input_and_warmup_seconds", MPI_Wtime() - placed}, {"matrix_bytes", residentBytes}}.dump() << std::endl;
    cudaEvent_t start{}, end{};
    Check(cudaEventCreate(&start));
    Check(cudaEventCreate(&end));
    const auto publication = resident.AcquireResidentRead();
    const auto view = publication.DeviceView();
    const auto mo = dm.Owned().Read(context), mg = dm.Ghost().Read(context);
    const auto co = du.Owned().Read(context), cg = du.Ghost().Read(context);
    const auto bc = db.values.Read(context);
    const auto starts = workspace.devicePeriodicVectorStarts.Read(context);
    auto status = workspace.deviceStatus.Write(context);
    auto output = workspace.temporary.Owned().Write(context);
    const D::DeviceWorkSelection selection{{}, metrics.NumCell(), false};
    const auto optionsFor = [](const J &row)
    {
        C::ReconstructionSweepLaunch options;
        options.threads = row.value("threads", 0);
        options.gridBlocks = row.value("gridBlocks", 0);
        return options;
    };
    const auto arithmeticFor = [](const J &row)
    {
        const auto kernel = row.at("kernel").get<std::string>();
        return kernel == "float-float"           ? D::ReconstructionSweepArithmetic::FloatFloat
               : kernel == "tensor"              ? D::ReconstructionSweepArithmetic::FP64TensorCore
               : kernel == "quantized-simt"      ? D::ReconstructionSweepArithmetic::FP32
               : kernel == "quantized-tensor"    ? D::ReconstructionSweepArithmetic::TF32
               : kernel == "fp32"                ? D::ReconstructionSweepArithmetic::FP32Reference
               : kernel == "fp32Accum"           ? D::ReconstructionSweepArithmetic::FP32FullAccumulation
               : kernel == "fp32FloatFloatAccum" ? D::ReconstructionSweepArithmetic::FP32FloatFloatAccumulation
               : kernel == "tf32"                ? D::ReconstructionSweepArithmetic::TF32Synchronous
               : kernel == "matrix-shared"       ? D::ReconstructionSweepArithmetic::FP64MatrixShared
                                                 : D::ReconstructionSweepArithmetic::FP64;
    };
    const auto launch = [&](const J &row)
    {
        const auto kernel = row.at("kernel").get<std::string>();
        if (kernel == "general")
            D::LaunchJacobi(context, view, selection, {starts.View()}, {mo.View(), mg.View()}, {co.View(), cg.View()}, bc.View(), output.View(), status.View().ConstView());
        else if (kernel == "matrix-shared")
            D::LaunchJacobiMatrixShared(context, view, selection, {starts.View()}, {mo.View(), mg.View()}, {co.View(), cg.View()}, bc.View(), output.View(), status.View().ConstView());
        else
        {
            DNDS_check_throw_info(kernel == "simt" || kernel == "tensor" || kernel == "float-float" || kernel == "quantized-simt" || kernel == "quantized-tensor" ||
                                      kernel == "fp32" || kernel == "fp32Accum" || kernel == "fp32FloatFloatAccum" || kernel == "tf32",
                                  "unknown retained sweep kernel");
            D::LaunchStagedJacobi(context, view, selection, {starts.View()}, {mo.View(), mg.View()}, {co.View(), cg.View()},
                                  bc.View(), output.View(), status.View().ConstView(), optionsFor(row), arithmeticFor(row));
            if (kernel == "float-float" || kernel == "quantized-simt" || kernel == "quantized-tensor")
                D::LaunchValidateJacobiResult(context, selection, output.View().ConstView(), status.View());
        }
    };
    for (const auto &row : catalog)
    {
        J record = row;
        record["representation"] = representation;
        record["subpartitions"] = partitions;
        record["device"] = properties.name;
        const auto kernel = row.at("kernel").get<std::string>();
        const bool narrow = kernel == "quantized-simt" || kernel == "quantized-tensor" || kernel == "fp32" ||
                            kernel == "fp32Accum" || kernel == "fp32FloatFloatAccum" || kernel == "tf32";
        const bool raw = representation == "raw-full" || representation == "raw-triangle";
        if ((raw && kernel != "general" && kernel != "simt") ||
            (representation == "quantized" && kernel != "general" && !narrow) ||
            (representation != "quantized" && narrow))
        {
            record["status"] = "unsupported_storage_fixture";
            std::cout << "VR_SWEEP " << record.dump() << std::endl;
            continue;
        }
        if ((row.at("kernel") == "tensor" && !D::FP64TensorCoreAvailable(device)) ||
            ((row.at("kernel") == "quantized-tensor" || row.at("kernel") == "tf32") && !D::TF32TensorCoreAvailable(device)))
        {
            record["status"] = "unsupported_tensor_device";
            std::cout << "VR_SWEEP " << record.dump() << std::endl;
            continue;
        }
        if (row.at("kernel") != "general" && row.at("kernel") != "matrix-shared")
        {
            const auto resource = D::InspectStagedJacobi(context, view, selection, {starts.View()}, {mo.View(), mg.View()},
                                                         {co.View(), cg.View()}, bc.View(), output.View(), status.View().ConstView(), optionsFor(row), arithmeticFor(row));
            record["resources"] = {{"shared_bytes", resource.sharedBytes}, {"registers_per_thread", resource.registersPerThread}, {"resident_blocks_per_sm", resource.activeBlocksPerSM}, {"grid_blocks", resource.gridBlocks}, {"threads_per_block", resource.threads}};
        }
        launch(row);
        Check(cudaStreamSynchronize(static_cast<cudaStream_t>(context.stream)));
        std::vector<double> actual(size_t(metrics.NumCell() * components * plan.NRecDOF()));
        Check(cudaMemcpy(actual.data(), output.View().data, actual.size() * sizeof(double), cudaMemcpyDeviceToHost));
        const auto expected = reference.Read();
        double error = 0;
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I j = 0; j < components * plan.NRecDOF(); ++j)
            {
                const auto value = actual[size_t(iCell * components * plan.NRecDOF() + j)];
                DNDS_check_throw_info(std::isfinite(value), "nonfinite sweep output");
                error = std::max(error, std::abs(value - expected.View()(iCell, j)) / (1 + std::abs(expected.View()(iCell, j))));
            }
        const double tolerance = row.value("tolerance", 2e-10);
        DNDS_check_throw_info(error <= tolerance, "sweep numerical comparison failed");
        record["tolerance"] = tolerance;
        std::vector<double> times;
        for (int trial = 0; trial < 3; ++trial)
        {
            Check(cudaEventRecord(start, static_cast<cudaStream_t>(context.stream)));
            for (int repeat = 0; repeat < repetitions; ++repeat)
                launch(row);
            Check(cudaEventRecord(end, static_cast<cudaStream_t>(context.stream)));
            Check(cudaEventSynchronize(end));
            float milliseconds = 0;
            Check(cudaEventElapsedTime(&milliseconds, start, end));
            times.push_back(milliseconds / repetitions);
        }
        auto sorted = times;
        std::sort(sorted.begin(), sorted.end());
        record["status"] = "verified";
        record["milliseconds"] = times;
        record["median_ms"] = sorted[1];
        record["maximum_relative_difference"] = error;
        std::cout << "VR_SWEEP " << record.dump() << std::endl;
    }
    Check(cudaEventDestroy(start));
    Check(cudaEventDestroy(end));
}
int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    try
    {
        Run(argc, argv);
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    DNDS::MPI::Finalize();
}
