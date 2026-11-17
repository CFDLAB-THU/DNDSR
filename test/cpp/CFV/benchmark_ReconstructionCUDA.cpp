// Actual resident VR wrappers, not synthetic layout arithmetic. CPU operations
// supply numerical oracles; local CPU time is not a cluster performance claim.
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "DNDS/Runtime/Communication/NCCLTransfer.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "../Geom/NativeMeshOverlapFixture.hpp"
#include <iostream>

namespace C = DNDS::CFV;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static void Fill(C::FVField &field, double offset)
{
    auto host = field.Owned().To(R::Device::Host());
    {
        auto w = host.Write();
        for (I iRow = 0; iRow < host.RowCount(); ++iRow)
            for (I j = 0; j < I(field.Width()); ++j)
                w.View()(iRow, j) = offset + .001 * ((iRow * 17 + j * 7) % 101);
    }
    field.Owned().CopyFrom(host, {field.Device(), nullptr}).Wait();
}
static double Error(const C::FVField &cpu, const C::FVField &gpu, DNDS::MPIInfo mpi)
{
    const auto host = gpu.Owned().To(R::Device::Host());
    const auto a = cpu.Owned().Read(), b = host.Read();
    double error = 0, result = 0;
    for (I iRow = 0; iRow < host.RowCount(); ++iRow)
        for (I j = 0; j < I(cpu.Width()); ++j)
            error = std::max(error, std::abs(a.View()(iRow, j) - b.View()(iRow, j)));
    MPI_Allreduce(&error, &result, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
    return result;
}
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    int result = 0;
    try
    {
        DNDS_check_throw_info(argc >= 5 && argc <= 8, "arguments: grid-width degree components repetitions [direct|packed|nccl] [fp64|async|pipelined|floatFloat|fp32|tf32] [all|aos]");
        const I width = std::stoll(argv[1]), components = std::stoll(argv[3]);
        const int degree = std::stoi(argv[2]), repeats = std::stoi(argv[4]);
        DNDS_check_throw_info(components > 0 && components <= 64 && repeats > 0, "invalid benchmark shape");
        const std::string transportName = argc >= 6 ? argv[5] : "direct", kernelName = argc >= 7 ? argv[6] : "fp64";
        const std::string layouts = argc >= 8 ? argv[7] : "all";
        DNDS_check_throw_info(layouts == "all" || layouts == "aos", "unknown benchmark layout selection");
        DNDS_check_throw_info(transportName == "direct" || transportName == "packed" || transportName == "nccl", "unknown transfer backend");
        DNDS_check_throw_info(kernelName == "fp64" || kernelName == "pipelined" || kernelName == "floatFloat" || kernelName == "async" ||
                                  kernelName == "fp32" || kernelName == "tf32",
                              "unknown reconstruction kernel");
        const auto transport = transportName == "direct" ? R::TransferTransport::Direct : R::TransferTransport::DevicePacked;
        const auto kernel = kernelName == "fp32" ? C::ReconstructionDeviceKernel::FP32 : kernelName == "tf32"     ? C::ReconstructionDeviceKernel::TF32TensorCore
                                                                                     : kernelName == "pipelined"  ? C::ReconstructionDeviceKernel::FP64Pipelined
                                                                                     : kernelName == "floatFloat" ? C::ReconstructionDeviceKernel::FloatFloatPipelined
                                                                                     : kernelName == "async"      ? C::ReconstructionDeviceKernel::FP64TensorCoreAsync
                                                                                                                  : C::ReconstructionDeviceKernel::FP64;
        DNDS_check_throw_info(kernel == C::ReconstructionDeviceKernel::FP64 || kernel == C::ReconstructionDeviceKernel::FP64Pipelined,
                              "this 2D benchmark cannot exercise C=5/M=19 comparison selectors; use cfv_benchmark_reconstruction_sweep");
        DNDS_check_throw_info(kernel == C::ReconstructionDeviceKernel::FP64 || layouts == "aos",
                              "optimized VR requires the explicit aos layout selection; use fp64 for generic layouts");
        const auto device = R::Device::CUDA(0);
        const auto context = R::ExecutionContext::CUDA(0);
        R::TransferOptions transferOptions;
        if (transportName == "nccl")
            transferOptions.provider = R::NCCLTransferProvider::Create(mpi.comm, device);
        auto measure = [&](auto &&action)
        {
            MPI_Barrier(mpi.comm);
            const auto start = MPI_Wtime();
            action();
            const double elapsed = MPI_Wtime() - start;
            double maximum = 0;
            MPI_Allreduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            return maximum;
        };
        std::shared_ptr<DNDS::Geom::UnstructuredMesh> mesh;
        const double meshTime = measure([&]
                                        { mesh = DNDS::Geom::OverlapTest::BuildNativeFixture(mpi, false, width).mesh; });
        std::optional<C::FiniteVolumePlan> metrics;
        const double metricTime = measure([&]
                                          { metrics = C::FiniteVolumePlan::ConstructMetrics(mesh); });
        C::ReconstructionSettings settings;
        settings.maxOrder = degree;
        std::optional<C::VariationalReconstructionPlan> plan;
        const double prepareTime = measure([&]
                                           { plan = C::VariationalReconstructionPlan::ConstructRecCoeff(*metrics, settings); });
        std::optional<C::VariationalReconstructionPlan> resident;
        C::ReconstructionResidencyOptions residency;
        if (kernelName == "fp32" || kernelName == "tf32")
            residency.matrixPrecision = C::ReconstructionMatrixPrecision::FP32;
        const double uploadTime = measure([&]
                                          { resident = plan->Place(device, context, residency); });
        const I recWidth = components * plan->NRecDOF();
        auto mean = metrics->BuildField(components);
        Fill(mean, 1);
        auto cpu = plan->BuildWorkspace(components);
        auto cpuCoefficients = plan->BuildCoefficients(components);
        auto bc = plan->BuildBoundaryValues(components);
        {
            auto write = bc.values.Write();
            for (I iRow = 0; iRow < bc.values.RowCount(); ++iRow)
                for (I j = 0; j < components; ++j)
                    write.View()(iRow, j) = 1 + .01 * j;
        }
        auto deviceBC = resident->PlaceBoundaryValues(bc);
        metrics->BindHalo(mean).Begin().Wait();
        for (auto kind : {R::RowLayoutKind::Table, R::RowLayoutKind::ColMajorTable, R::RowLayoutKind::TiledColMajorTable})
        {
            if (layouts == "aos" && kind != R::RowLayoutKind::Table)
                continue;
            const auto shape = [&](I count)
            { return kind == R::RowLayoutKind::Table ? R::RowLayout::Dense(0, count) : kind == R::RowLayoutKind::ColMajorTable ? R::RowLayout::ColMajor(0, count, {32, 1})
                                                                                                                               : R::RowLayout::TiledColMajor(0, count, 32); };
            auto gpuMean = metrics->BuildField(shape(components), resident->Device());
            Fill(gpuMean, 1);
            auto gpu = resident->BuildWorkspace(components, shape(recWidth));
            auto gpuCoefficients = resident->BuildCoefficients(components, shape(recWidth));
            Fill(cpuCoefficients, .02);
            Fill(gpuCoefficients, .02);
            auto gpuMeanHalo = resident->BindHalo(gpuMean, transport, transferOptions), gpuHalo = resident->BindHalo(gpuCoefficients, transport, transferOptions), cpuHalo = metrics->BindHalo(cpuCoefficients);
            gpuMeanHalo.Begin(context).Wait();
            cpuHalo.Begin().Wait();
            gpuHalo.Begin(context).Wait();
            plan->DoReconstructionIter(mean, cpuCoefficients, cpu, bc);
            resident->DoReconstructionIter(gpuMean, gpuCoefficients, gpu, deviceBC, kernel);
            const double jacobiError = Error(cpuCoefficients, gpuCoefficients, mpi);
            for (int order = degree; order >= 1; --order)
            {
                cpuHalo.Begin().Wait();
                gpuHalo.Begin(context).Wait();
                plan->DoLimiterWBAP_3Order(cpuCoefficients, cpu, order);
                resident->DoLimiterWBAP_3Order(gpuCoefficients, gpu, order);
            }
            const double limiterError = Error(cpuCoefficients, gpuCoefficients, mpi);
            const double tolerance = kernelName == "tf32" ? 3e-3 : kernelName == "fp32" ? 2e-5
                                                                                        : 1e-8;
            DNDS_check_throw_info(jacobiError < tolerance && limiterError < tolerance, "GPU benchmark numerical oracle failed");
            gpuHalo.Begin(context).Wait();
            R::EagerEngine engine;
            auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            auto commStream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
            for (int warm = 0; warm < 3; ++warm)
                resident->SubmitReconstructionIterTo(engine, stream, gpuMean, gpuCoefficients, gpu, deviceBC, C::ReconstructionRegion::All, kernel);
            engine.Synchronize();
            // A separate, untimed overlap probe extends real owned-only work.
            // Observe a started but unfinished compute prefix at MPI completion;
            // do not mix these extra event/launch costs into operator timings.
            int overlapStarted = 0, overlapPending = 0;
            if (mpi.size > 1)
            {
                MPI_Barrier(mpi.comm);
                R::TransferOperation transfer;
                if (transferOptions.provider)
                    gpuHalo.Enqueue(engine, commStream);
                else
                    transfer = gpuHalo.Begin(context);
                const auto transferred = transferOptions.provider ? commStream.Record() : R::StreamEvent{};
                const auto begin = stream.Record();
                for (int repeat = 0; repeat < 32; ++repeat)
                    resident->SubmitReconstructionIterTo(engine, stream, gpuMean, gpuCoefficients, gpu, deviceBC, C::ReconstructionRegion::Core, kernel);
                const auto end = stream.Record();
                if (transferOptions.provider)
                    transferred.WaitHost();
                else
                    transfer.Wait();
                overlapStarted = int(begin.Ready());
                overlapPending = int(!end.Ready());
                engine.Synchronize();
            }
            for (int trial = 0; trial < 3; ++trial)
            {
                const double jacobi = measure([&]
                                              {for(int step=0;step<repeats;++step)resident->SubmitReconstructionIterTo(engine,stream,gpuMean,gpuCoefficients,gpu,deviceBC,C::ReconstructionRegion::All,kernel);engine.Synchronize(); }) /
                                      repeats;
                const double wbap = measure([&]
                                            {for(int step=0;step<repeats;++step)resident->SubmitLimiterWBAP_3OrderTo(engine,stream,gpuCoefficients,gpu,degree);engine.Synchronize(); }) /
                                    repeats;
                const double comm = measure([&]
                                            {for(int step=0;step<repeats;++step) { if(transferOptions.provider) gpuHalo.Enqueue(engine,commStream); else gpuHalo.Begin(context).Wait(); } engine.Synchronize(); }) /
                                    repeats;
                const double integrated = measure([&]
                                                  {for(int step=0;step<repeats;++step)
                {
                    R::TransferOperation exchange;
                    if(transferOptions.provider) { commStream.WaitFor(stream.Record()); gpuHalo.Enqueue(engine,commStream); }
                    else exchange=gpuHalo.Begin(context);
                    resident->SubmitReconstructionIterTo(engine,stream,gpuMean,gpuCoefficients,gpu,deviceBC,C::ReconstructionRegion::Core,kernel);
                    if(transferOptions.provider) stream.WaitFor(commStream.Record()); else exchange.Wait();
                    resident->SubmitReconstructionIterTo(engine,stream,gpuMean,gpuCoefficients,gpu,deviceBC,C::ReconstructionRegion::HaloDependent,kernel);
                    resident->SubmitCommitReconstruction(engine,stream,gpuCoefficients,gpu);
                    if(!transferOptions.provider) engine.Synchronize();
                } engine.Synchronize(); }) /
                                          repeats;
                resident->CheckStatus(gpu);
                if (mpi.rank == 0)
                    std::cout << "CUDA_VR_BENCH " << nlohmann::ordered_json{{"np", mpi.size}, {"cells", width * width}, {"degree", degree}, {"components", components}, {"layout", int(kind)}, {"provider", transferOptions.provider ? "nccl" : "mpi"}, {"kernel", kernelName}, {"transport", transport == R::TransferTransport::DevicePacked ? "packed" : "direct"}, {"trial", trial}, {"mesh", meshTime}, {"metrics", metricTime}, {"prepare", prepareTime}, {"upload", uploadTime}, {"jacobi", jacobi}, {"wbap_highest_order", wbap}, {"communication", comm}, {"integrated", integrated}, {"core_prefix_started_at_transfer_completion", overlapStarted}, {"core_prefix_pending_at_transfer_completion", overlapPending}, {"jacobi_error", jacobiError}, {"limiter_error", limiterError}}.dump() << std::endl;
            }
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << std::endl;
        result = 1;
    }
    MPI_Finalize();
    return result;
}
