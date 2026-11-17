// Bounded actual-VR exploration. Setup, fixed apply, commit and contraction
// remain separate observations; this is not a new solver convergence gate.
#include "CFV/Plans/ReconstructionAMGPlan.hpp"
#include "DNDS/Algebra/AmgXDiagnostics.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "Geom/IO/NativeMeshCGNS.hpp"
#include <fstream>
#include <iostream>

namespace C = DNDS::CFV;
namespace R = DNDS::Runtime;
namespace G = DNDS::Geom;
namespace A = DNDS::Algebra;
using I = DNDS::index;
using J = nlohmann::ordered_json;

static void Run(int argc, char **argv)
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    DNDS_check_throw_info(mpi.size == 1 && argc == 7,
                          "usage: vr_amg periodic-cube.cgns degree upload|native raw-full|precomputed fixed_updates fixed_cycles (np1)");
    const int degree = std::stoi(argv[2]), updates = std::stoi(argv[5]), cycles = std::stoi(argv[6]);
    const std::string storage = argv[3], representation = argv[4];
    DNDS_check_throw_info(degree >= 1 && degree <= 3 && updates > 0 && cycles > 0 &&
                              (storage == "upload" || storage == "native") && (representation == "raw-full" || representation == "precomputed"),
                          "invalid VR AMG exploration controls");
    G::NativePeriodicImportOptions periodic;
    for (int d = 0; d < 3; ++d)
    {
        periodic.frames.translation[d + 1].map().setZero();
        periodic.frames.translation[d + 1].map()[d] = 6.2831853071795864769;
    }
    auto mesh = G::UnstructuredMesh::FromNative(mpi, G::ReadNativeCGNS(mpi, argv[1], 3, G::FBC_Name_2_ID_Default, 0, {1, true}, periodic));
    mesh->PrepareNativeTopology();
    C::FiniteVolumeMetricOptions metricOptions;
    metricOptions.integrationOrder = 5;
    metricOptions.collectivePreflight = false;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh, metricOptions);
    C::ReconstructionSettings settings;
    settings.maxOrder = degree;
    settings.intOrder = 5;
    settings.cacheDiffBase = false;
    settings.SORInstead = false;
    settings.jacobiRelax = 1;
    settings.storage.representation = representation == "raw-full" ? "raw" : "precomputed";
    settings.storage.faceCoupling = representation == "raw-full";
    const double prepareStart = MPI_Wtime();
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings, C::ReconstructionCapabilities::ForSweeps(false));
    const double prepareSeconds = MPI_Wtime() - prepareStart;
    constexpr I components = 5;
    const I width = components * plan.NRecDOF();
    auto mean = metrics.BuildField(components), initial = plan.BuildCoefficients(components);
    auto bc = plan.BuildBoundaryValues(components);
    {
        auto m = mean.Owned().Write(), x = initial.Owned().Write(), b = bc.values.Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            for (I j = 0; j < components; ++j)
                m.View()(iCell, j) = 1 + .01 * std::sin(.017 * iCell + .13 * j);
            for (I j = 0; j < width; ++j)
                x.View()(iCell, j) = 0;
        }
        for (I q = 0; q < bc.values.RowCount(); ++q)
            for (I j = 0; j < components; ++j)
                b.View()(q, j) = 1;
    }
    metrics.BindHalo(mean).Begin().Wait();
    metrics.BindHalo(initial).Begin().Wait();
    const auto device = R::Device::CUDA(0);
    R::EagerEngine engine;
    const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    const auto context = stream.Context();
    auto deviceMean = metrics.BuildField(R::RowLayout::Dense(0, components), device);
    auto input = metrics.BuildField(R::RowLayout::Dense(0, width), device);
    auto output = metrics.BuildField(R::RowLayout::Dense(0, width), device);
    deviceMean.Owned().CopyFrom(mean.Owned(), context).Wait();
    deviceMean.Ghost().CopyFrom(mean.Ghost(), context).Wait();
    input.Owned().CopyFrom(initial.Owned(), context).Wait();
    input.Ghost().CopyFrom(initial.Ghost(), context).Wait();
    auto cpuWork = plan.BuildWorkspace(components);
    const auto residual = [&]
    {
        initial.Owned().CopyFrom(input.Owned(), context).Wait();
        metrics.BindHalo(initial).Begin().Wait();
        plan.DoReconstructionIterTo(mean, initial, cpuWork, bc);
        const auto x = initial.Owned().Read(), y = cpuWork.temporary.Owned().Read();
        double sum = 0;
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I j = 0; j < width; ++j)
            {
                const double value = y.View()(iCell, j) - x.View()(iCell, j);
                sum += value * value;
            }
        return std::sqrt(sum);
    };
    const double initialResidual = residual();
    // Measure the ordinary production sweep separately and release its GPU
    // matrices before creating AMG. Do not retain two native representations
    // to hide the memory cost of an adapter. Both use the same fixed inputs.
    {
        auto resident = plan.Place(device, context);
        auto sweep = resident.BuildWorkspace(components);
        const auto placedBC = resident.PlaceBoundaryValues(bc);
        for (int iteration = 0; iteration < updates; ++iteration)
        {
            const double start = MPI_Wtime();
            resident.SubmitReconstructionIterTo(engine, stream, deviceMean, input, sweep, placedBC,
                                                C::ReconstructionRegion::All, C::ReconstructionDeviceKernel::FP64Pipelined);
            engine.Synchronize();
            const double applySeconds = MPI_Wtime() - start;
            const double commitStart = MPI_Wtime();
            resident.SubmitCommitReconstruction(engine, stream, input, sweep);
            engine.Synchronize();
            const double commitSeconds = MPI_Wtime() - commitStart;
            const double norm = residual();
            std::cout << "VR_JACOBI " << J{{"cells", metrics.NumCell()}, {"modes", plan.NRecDOF()}, {"iteration", iteration}, {"apply_seconds", applySeconds}, {"commit_seconds", commitSeconds}, {"residual", norm}, {"residual_ratio", norm / initialResidual}}.dump() << std::endl;
        }
    }
    {
        auto write = initial.Owned().Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I j = 0; j < width; ++j)
                write.View()(iCell, j) = 0;
    }
    input.Owned().CopyFrom(initial.Owned(), context).Wait();
    A::AmgXSettings amgSettings;
    amgSettings.cycles = cycles;
    const double setupStart = MPI_Wtime();
    auto amg = C::ReconstructionAMGPlan::Prepare(plan, context,
                                                 storage == "native" ? C::ReconstructionAMGStorage::NativeStorage : C::ReconstructionAMGStorage::PublicUpload, amgSettings);
    const double setupSeconds = MPI_Wtime() - setupStart;
    auto work = amg.BuildWorkspace(components);
    const auto deviceBC = amg.PlaceBoundaryValues(bc);
    const J common{{"cells", metrics.NumCell()}, {"modes", plan.NRecDOF()}, {"components", components}, {"storage", storage}, {"representation", representation}, {"cycles", cycles}, {"vr_prepare_seconds", prepareSeconds}, {"amg_setup_seconds", setupSeconds}};
    auto receipt = common;
    receipt["phase"] = "setup";
    receipt["memory"] = J::parse(A::AmgXMemoryReport(amg.MemorySnapshot()));
    receipt["residual"] = initialResidual;
    std::cout << "VR_AMG " << receipt.dump() << std::endl;
    for (int iteration = 0; iteration < updates; ++iteration)
    {
        const double start = MPI_Wtime();
        amg.SubmitDefectCorrection(engine, stream, deviceMean, input, deviceBC, output, work);
        engine.Synchronize();
        const double applySeconds = MPI_Wtime() - start;
        const double commitStart = MPI_Wtime();
        input.Owned().CopyFrom(output.Owned(), context).Wait();
        const double commitSeconds = MPI_Wtime() - commitStart;
        const double norm = residual();
        DNDS_check_throw_info(std::isfinite(norm), "nonfinite VR AMG residual");
        receipt = common;
        receipt["phase"] = "fixed_update";
        receipt["iteration"] = iteration;
        receipt["apply_seconds"] = applySeconds;
        receipt["commit_seconds"] = commitSeconds;
        receipt["residual"] = norm;
        receipt["residual_ratio"] = norm / initialResidual;
        std::cout << "VR_AMG " << receipt.dump() << std::endl;
    }
    engine.Close();
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
        // No recovery from a failure in collectively ordered preparation.
        std::cerr << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    DNDS::MPI::Finalize();
}
