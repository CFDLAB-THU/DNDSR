#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Runtime/Execution/EagerCopy.hpp"
#include "DNDS/Runtime/Storage/AllocationResource.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <thread>
#include <chrono>

namespace R = DNDS::Runtime;

TEST_CASE("CUDA eager operands retain whole-array placement through kernel submission")
{
    R::EagerEngine engine;
    const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    R::Array<double> source(R::RowLayout::Dense(19, 5));
    {
        auto write = source.Write();
        for (DNDS::index iRow = 0; iRow < 19; ++iRow)
            for (DNDS::index j = 0; j < 5; ++j)
                write.View()(iRow, j) = 10 * iRow + j;
    }
    auto device = source.To(R::Device::CUDA(0));
    bool checked = false;
    R::SubmitArrays(engine, stream, [&checked](R::ExecutionContext context, auto input)
                    {
                        input.ValidateDevice(context.device);
                        CHECK(input.GetDevice() == R::Device::CUDA(0));
                        CHECK(input.ConstView().GetDevice() == context.device);
                        CHECK_THROWS(input.ValidateDevice(R::Device::Host()));
                        checked = true; }, R::Read(device));
    engine.Close();
    CHECK(checked);
    auto result = device.To(R::Device::Host());
    auto read = result.Read();
    CHECK(read.View().GetDevice() == R::Device::Host());
    CHECK(read.View()(18, 4) == 184);
}

struct Gate
{
    std::shared_ptr<std::atomic<bool>> open = std::make_shared<std::atomic<bool>>(false);
    ~Gate() { *open = true; }
    void Enqueue(R::EagerEngine &engine, const R::ExecutionStream &stream)
    {
        engine.Submit(stream, [owner = open](R::ExecutionContext context)
                      {
            auto status = cudaLaunchHostFunc(static_cast<cudaStream_t>(context.stream), [](void *p)
            {
                auto &value = *static_cast<std::atomic<bool> *>(p);
                while (!value.load()) std::this_thread::sleep_for(std::chrono::microseconds(50));
            }, owner.get());
            DNDS_check_throw_info(status == cudaSuccess, "test gate launch failed"); });
    }
};

__global__ void Add(R::ArrayView<double> u, double value)
{
    for (size_t row = blockIdx.x; row < u.RowCount(); row += gridDim.x)
        for (size_t col = threadIdx.x; col < u.RowSize(row); col += blockDim.x)
            u(row, col) += value;
}
__global__ void Scale(R::ArrayView<const double> in, R::ArrayView<double> out, double factor)
{
    for (size_t row = blockIdx.x; row < out.RowCount(); row += gridDim.x)
        for (size_t col = threadIdx.x; col < out.RowSize(row); col += blockDim.x)
            out(row, col) = in(row, col) * factor;
}
__global__ void CoreStencil(R::ArrayView<const int64_t> neighbors, R::ArrayView<const double> owned, R::ArrayView<double> out)
{
    for (size_t row = blockIdx.x; row < neighbors.RowCount(); row += gridDim.x)
        for (size_t col = threadIdx.x; col < out.RowSize(row); col += blockDim.x)
        {
            double sum = 0;
            for (size_t j = 0; j < neighbors.RowSize(row); ++j)
                sum += owned(neighbors(row, j), col);
            out(row, col) = sum;
        }
}
__global__ void RingUpdate(R::ArrayView<const double> ghost, R::ArrayView<double> out)
{
    for (size_t row = blockIdx.x; row < ghost.RowCount(); row += gridDim.x)
        for (size_t col = threadIdx.x; col < ghost.RowSize(row); col += blockDim.x)
            out(out.RowCount() - ghost.RowCount() + row, col) = ghost(row, col);
}

__device__ bool Rendezvous(int *flag, int *failure)
{
    const auto start = clock64();
    while (atomicAdd(flag, 0) == 0)
        if (clock64() - start > 5000000000ULL)
        {
            atomicExch(failure, 1);
            return false;
        }
    return true;
}

__global__ void ConcurrentMock(R::ArrayView<const double> owned, R::ArrayView<double> output,
                               R::ArrayView<int> flags, bool communication)
{
    // One resident block per stream, no oversubscribed-grid barrier. Atomic
    // flags are intentional inter-stream synchronization solely for the test.
    if (threadIdx.x == 0)
    {
        atomicExch(&flags(0, communication ? 0 : 1), 1);
        Rendezvous(&flags(0, communication ? 1 : 0), &flags(0, 3));
    }
    __syncthreads();
    for (size_t i = threadIdx.x; i < output.RowCount() * output.width; i += blockDim.x)
    {
        const auto row = i / output.width, col = i % output.width;
        output(row, col) = communication ? owned(13 * row, col) : 2 * owned(row, col);
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        if (communication)
            Rendezvous(&flags(0, 2), &flags(0, 3));
        else
            atomicExch(&flags(0, 2), 1);
    }
}
static void LaunchAdd(R::ExecutionContext context, R::ArrayView<double> u)
{
    Add<<<7, 32, 0, static_cast<cudaStream_t>(context.stream)>>>(u, 1);
    DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "test Add launch failed");
}

__global__ void RestrictLevel(R::ArrayView<const double> fine, R::ArrayView<double> coarse)
{
    for (size_t row = blockIdx.x; row < coarse.RowCount(); row += gridDim.x)
        for (size_t col = threadIdx.x; col < coarse.RowSize(row); col += blockDim.x)
        {
            const auto left = 2 * row, right = left + 1 < fine.RowCount() ? left + 1 : left;
            coarse(row, col) = 0.25 * (fine(left, col) + fine(right, col));
        }
}

__global__ void ProlongLevel(R::ArrayView<const double> coarse, R::ArrayView<double> fine)
{
    for (size_t row = blockIdx.x; row < fine.RowCount(); row += gridDim.x)
        for (size_t col = threadIdx.x; col < fine.RowSize(row); col += blockDim.x)
            fine(row, col) += coarse(row / 2, col);
}

static void PreloadKernels()
{
    // CUDA lazy first-use loading may synchronize the context. Load before
    // deliberately blocking a stream; otherwise the test gate and first launch
    // can deadlock independently of the engine's ordering.
    cudaFuncAttributes attributes{};
    DNDS_check_throw_info(cudaFuncGetAttributes(&attributes, Add) == cudaSuccess, "test Add preload failed");
    DNDS_check_throw_info(cudaFuncGetAttributes(&attributes, Scale) == cudaSuccess, "test Scale preload failed");
    DNDS_check_throw_info(cudaFuncGetAttributes(&attributes, CoreStencil) == cudaSuccess, "test stencil preload failed");
    DNDS_check_throw_info(cudaFuncGetAttributes(&attributes, RingUpdate) == cudaSuccess, "test ring preload failed");
    DNDS_check_throw_info(cudaFuncGetAttributes(&attributes, ConcurrentMock) == cudaSuccess, "test concurrent preload failed");
    DNDS_check_throw_info(cudaFuncGetAttributes(&attributes, RestrictLevel) == cudaSuccess, "test restriction preload failed");
    DNDS_check_throw_info(cudaFuncGetAttributes(&attributes, ProlongLevel) == cudaSuccess, "test prolongation preload failed");
}

TEST_CASE("CUDA mutation epochs precede completion and survive stream retirement")
{
    PreloadKernels();
    R::Array<double> host(R::RowLayout::Dense(17, 5));
    {
        auto write = host.Write();
        for (DNDS::index iRow = 0; iRow < 17; ++iRow)
            for (DNDS::index j = 0; j < 5; ++j)
                write.View()(iRow, j) = 1.;
    }
    auto values = host.To(R::Device::CUDA(0));
    auto alias = values;
    R::EagerEngine engine;
    const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    Gate gate; // Opens before engine destruction, including assertion unwinding.
    const auto version = values.Version(), epoch = values.MutationEpoch();
    gate.Enqueue(engine, stream);
    R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto output)
                    { LaunchAdd(context, output); }, R::ReadWrite(values));
    CHECK(alias.MutationEpoch() == epoch + 1);
    CHECK(alias.Version() == version);
    CHECK_THROWS(alias.Read(R::ExecutionContext::CUDA(0)));
    *gate.open = true;
    engine.Synchronize();
    CHECK(alias.MutationEpoch() == epoch + 1);
    CHECK(alias.Version() == version + 1);
    const auto result = values.To(R::Device::Host());
    CHECK(result.Read().View()(16, 4) == 2.);
}

TEST_CASE("Closing mixed CPU CUDA execution retains temporary operands and observes completed events")
{
    PreloadKernels();
    struct OpenGate final : R::ExecutionProgress
    {
        std::shared_ptr<std::atomic<bool>> open;
        explicit OpenGate(std::shared_ptr<std::atomic<bool>> value) : open(std::move(value)) {}
        bool Poll() override
        {
            *open = true;
            return true;
        }
    };
    Gate gate;
    R::EagerEngine engine(1);
    auto cpu = engine.CreateCPUStream(); // Drain order must not block this waiter.
    auto launcher = std::make_shared<R::CUDALauncher>(0);
    auto gpu = engine.CreateStream(launcher);
    R::Array<double> host(R::RowLayout::Dense(17, 5));
    {
        auto write = host.Write();
        for (DNDS::index iRow = 0; iRow < 17; ++iRow)
            for (DNDS::index j = 0; j < 5; ++j)
                write.View()(iRow, j) = 1;
    }
    auto temporary = host.To(R::Device::CUDA(0));
    R::Array<double> output(host.Layout(), R::Device::CUDA(0));
    auto copyInput = host.To(R::Device::CUDA(0));
    R::Array<double> copyOutput(host.Layout(), R::Device::CUDA(0));
    std::weak_ptr<const R::ArrayStorage> retained = temporary.SharedStorage();
    engine.RegisterProgress(std::make_shared<OpenGate>(gate.open));
    gate.Enqueue(engine, gpu);
    auto unusedInitialization = engine.AllocateBuffer(gpu, 128);
    unusedInitialization = {}; // Initialization itself must retain its allocation.
    {
        auto source = copyInput.Storage(), destination = copyOutput.Storage();
        R::CopyAsync(engine, gpu, source, destination);
    }
    copyInput = {}; // The DMA operation owns its source until completion.
    R::SubmitArrays(engine, gpu, LaunchAdd, R::ReadWrite(temporary));
    R::SubmitArrays(engine, gpu, [](R::ExecutionContext context, auto input, auto out)
                    { Scale<<<7, 32, 0, static_cast<cudaStream_t>(context.stream)>>>(input, out, 2); }, R::Read(temporary), R::Write(output));
    const auto event = gpu.Record();
    temporary = {};
    CHECK_FALSE(retained.expired());
    cpu.WaitFor(event);
    std::atomic<bool> observed{false};
    engine.Submit(cpu, [&](R::ExecutionContext)
                  { observed = true; });
    engine.Close();
    launcher.reset(); // Completed prefix is still observable without its stream.
    CHECK(observed.load());
    CHECK(retained.expired());
    CHECK(event.Ready());
    CHECK_THROWS(gpu.Record());
    const auto result = output.To(R::Device::Host());
    const auto read = result.Read();
    for (DNDS::index iRow = 0; iRow < 17; ++iRow)
        for (DNDS::index j = 0; j < 5; ++j)
            CHECK(read.View()(iRow, j) == 4);
    const auto copied = copyOutput.To(R::Device::Host());
    const auto copyRead = copied.Read();
    for (DNDS::index iRow = 0; iRow < 17; ++iRow)
        for (DNDS::index j = 0; j < 5; ++j)
            CHECK(copyRead.View()(iRow, j) == 1);
}

TEST_CASE("CUDA dynamic multilevel temporaries join independent readers before final stream release")
{
    PreloadKernels();
    for (size_t depth : {size_t(3), size_t(4), size_t(5)})
    {
        R::EagerEngine engine;
        auto first = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
        auto second = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
        auto comm = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
        R::Array<double> host(R::RowLayout::Dense(257, 7, 9));
        {
            auto w = host.Write();
            for (size_t i = 0; i < host.RowCount(); ++i)
                for (size_t j = 0; j < 7; ++j)
                    w.At(i, j) = j + 1;
        }
        auto seed = host.To(R::Device::CUDA(0));
        std::vector<R::Array<double>> outputs, diagnostics;
        size_t coarseRows = 257;
        for (size_t i = 1; i < depth; ++i)
            coarseRows = (coarseRows + 1) / 2;
        for (int cycle = 0; cycle < 3; ++cycle)
        {
            outputs.emplace_back(seed.Layout(), R::Device::CUDA(0));
            diagnostics.emplace_back(R::RowLayout::Dense(coarseRows, 7, 10), R::Device::CUDA(0));
        }
        Gate gate;
        gate.Enqueue(engine, first);
        for (int cycle = 0; cycle < 3; ++cycle)
        {
            std::vector<R::Array<double>> levels;
            levels.push_back(R::AllocateArrayAsync<double>(engine, first, seed.Layout()));
            R::SubmitArrays(engine, first, [cycle](R::ExecutionContext c, auto in, auto out)
                            {
                Scale<<<7, 32, 0, static_cast<cudaStream_t>(c.stream)>>>(in, out, cycle + 1);
                DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "cycle seed launch failed"); }, R::Read(seed), R::Write(levels[0]));
            for (size_t level = 1; level < depth; ++level)
            {
                auto producer = level % 2 ? first : second;
                auto consumer = level % 2 ? second : first;
                consumer.WaitFor(producer.Record());
                levels.push_back(R::AllocateArrayAsync<double>(engine, consumer,
                                                               R::RowLayout::Dense((levels.back().RowCount() + 1) / 2, 7, 9)));
                R::SubmitArrays(engine, consumer, [](R::ExecutionContext c, auto fine, auto coarse)
                                {
                    RestrictLevel<<<7, 32, 0, static_cast<cudaStream_t>(c.stream)>>>(fine, coarse);
                    DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "restriction launch failed"); }, R::Read(levels[level - 1]), R::Write(levels[level]));
            }
            auto coarsest = (depth - 1) % 2 ? second : first;
            comm.WaitFor(coarsest.Record());
            R::SubmitArrays(engine, comm, [](R::ExecutionContext c, auto in, auto out)
                            {
                Scale<<<3, 32, 0, static_cast<cudaStream_t>(c.stream)>>>(in, out, 2);
                DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "diagnostic launch failed"); }, R::Read(levels.back()), R::Write(diagnostics[cycle]));
            for (size_t level = depth - 1; level > 0; --level)
            {
                auto producer = level % 2 ? second : first;
                auto consumer = level % 2 ? first : second;
                consumer.WaitFor(producer.Record());
                R::SubmitArrays(engine, consumer, [](R::ExecutionContext c, auto coarse, auto fine)
                                {
                    ProlongLevel<<<7, 32, 0, static_cast<cudaStream_t>(c.stream)>>>(coarse, fine);
                    DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "prolongation launch failed"); }, R::Read(levels[level]), R::ReadWrite(levels[level - 1]));
                auto alias = levels[level];
                // Non-final decrement on producer; final decrement on consumer.
                engine.ReleaseOn(std::move(levels[level]), producer);
                if (level == depth - 1)
                    consumer.WaitFor(comm.Record());
                engine.ReleaseOn(std::move(alias), consumer);
            }
            {
                auto destination = outputs[cycle].Storage();
                R::CopyAsync(engine, first, levels[0].Storage(), destination);
            }
            engine.ReleaseOn(std::move(levels[0]), first);
        }
        auto finished = first.Record();
        CHECK_FALSE(finished.Ready());
        CHECK_THROWS(outputs.back().Read(first.Context()));
        *gate.open = true;
        engine.Synchronize();
        for (int cycle = 0; cycle < 3; ++cycle)
        {
            auto result = outputs[cycle].To(R::Device::Host()), observed = diagnostics[cycle].To(R::Device::Host());
            auto r = result.Read(), d = observed.Read();
            const double factor = 2.0 - 1.0 / (size_t(1) << (depth - 1));
            for (size_t i = 0; i < result.RowCount(); ++i)
                for (size_t j = 0; j < 7; ++j)
                    CHECK(r.At(i, j) == factor * (j + 1) * (cycle + 1));
            for (size_t i = 0; i < observed.RowCount(); ++i)
                for (size_t j = 0; j < 7; ++j)
                    CHECK(d.At(i, j) == double((j + 1) * (cycle + 1)) / (size_t(1) << (depth - 2)));
        }
    }
}

TEST_CASE("CUDA mock communication and core kernels are simultaneously resident")
{
    cudaDeviceProp properties{};
    REQUIRE(cudaGetDeviceProperties(&properties, 0) == cudaSuccess);
    REQUIRE(properties.concurrentKernels);
    PreloadKernels();
    for (size_t width : {size_t(5), size_t(6), size_t(7), size_t(13), size_t(114)})
    {
        R::EagerEngine engine;
        auto compute = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
        auto comm = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
        R::Array<double> host(R::RowLayout::Dense(257, width, width + 3));
        {
            auto w = host.Write();
            for (size_t i = 0; i < 257; ++i)
                for (size_t j = 0; j < width; ++j)
                    w.At(i, j) = 10 * i + j;
        }
        auto owned = host.To(R::Device::CUDA(0));
        R::Array<double> ghost(R::RowLayout::Dense(19, width, width + 2), R::Device::CUDA(0));
        R::Array<double> output(host.Layout(), R::Device::CUDA(0));
        R::Array<int> flags(R::RowLayout::Dense(1, 4), R::Device::CUDA(0));
        auto launch = [](bool communication)
        {
            return [communication](R::ExecutionContext context, auto u, auto out, auto f)
            {
                ConcurrentMock<<<1, 128, 0, static_cast<cudaStream_t>(context.stream)>>>(u, out, f, communication);
                DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "concurrent mock launch failed");
            };
        };
        R::SubmitArrays(engine, comm, launch(true), R::Read(owned), R::Write(ghost), R::ReadWrite(flags));
        R::SubmitArrays(engine, compute, launch(false), R::Read(owned), R::Write(output), R::ReadWrite(flags));
        compute.WaitFor(comm.Record());
        R::SubmitArrays(engine, compute, [](R::ExecutionContext context, auto g, auto out)
                        { RingUpdate<<<3, 32, 0, static_cast<cudaStream_t>(context.stream)>>>(g, out); }, R::Read(ghost), R::ReadWrite(output));
        engine.Synchronize();
        auto status = flags.To(R::Device::Host());
        CHECK(status.Read().At(0, 3) == 0); // Serialization would time out a rendezvous.
        CHECK(status.Read().At(0, 2) == 1);
        auto result = output.To(R::Device::Host());
        auto read = result.Read();
        for (size_t i = 0; i < 257; ++i)
            for (size_t j = 0; j < width; ++j)
                CHECK(read.At(i, j) == (i < 238 ? 20 * i + 2 * j : 130 * (i - 238) + j));
    }
}

TEST_CASE("CUDA missing eager join returns stale values while ordered consumption is correct")
{
    R::EagerEngine engine;
    auto compute = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    auto comm = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    R::Array<double> ghost(R::RowLayout::Dense(19, 7, 9), R::Device::CUDA(0));
    R::Array<double> stale(ghost.Layout(), R::Device::CUDA(0)), correct(ghost.Layout(), R::Device::CUDA(0));
    PreloadKernels();
    Gate gate;
    gate.Enqueue(engine, comm);
    R::SubmitArrays(engine, comm, LaunchAdd, R::Write(ghost));
    auto consume = [](R::ExecutionContext context, auto g, auto out)
    { Scale<<<3, 32, 0, static_cast<cudaStream_t>(context.stream)>>>(g, out, 2); };
    R::SubmitArrays(engine, compute, consume, R::Read(ghost), R::Write(stale));
    auto copied = compute.Record();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!copied.Ready() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    const bool readBeforeWrite = copied.Ready();
    CHECK_THROWS(ghost.Read(comm.Context()));
    *gate.open = true; // Opens only after read completion: no device data race.
    REQUIRE(readBeforeWrite);
    compute.WaitFor(comm.Record());
    R::SubmitArrays(engine, compute, consume, R::Read(ghost), R::Write(correct));
    engine.Synchronize();
    auto badHost = stale.To(R::Device::Host()), goodHost = correct.To(R::Device::Host());
    auto bad = badHost.Read(), good = goodHost.Read();
    CHECK(stale.IsValid());
    for (size_t i = 0; i < 19; ++i)
        for (size_t j = 0; j < 7; ++j)
        {
            CHECK(bad.At(i, j) == 0);
            CHECK(good.At(i, j) == 2);
        }
}

TEST_CASE("Checked CUDA eagerly queues dependent kernels and rejects missing prefixes and free joins")
{
    auto tracking = std::make_shared<R::AccessTrackingContext>();
    R::EagerEngine engine(0, tracking);
    auto producer = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    auto consumer = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    R::Array<double> output(R::RowLayout::Dense(19, 7, 9), R::Device::CUDA(0));
    PreloadKernels();
    Gate gate;
    gate.Enqueue(engine, producer);
    const auto stale = producer.Record();
    auto temporary = R::AllocateArrayAsync<double>(engine, producer, R::RowLayout::Dense(19, 7, 9));
    consumer.WaitFor(stale);
    const auto consume = [](R::ExecutionContext context, auto input, auto result)
    {
        Scale<<<3, 32, 0, static_cast<cudaStream_t>(context.stream)>>>(input, result, 3);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "checked test Scale launch failed");
    };
    // Initialization itself is a pending writer, not ready merely because its
    // CPU handle exists. The old event precedes both initialization and Add.
    CHECK_THROWS(R::SubmitArrays(engine, consumer, consume, R::Read(temporary), R::Write(output)));
    R::SubmitArrays(engine, producer, LaunchAdd, R::ReadWrite(temporary));
    const auto ready = producer.Record();
    consumer.WaitFor(ready);
    R::SubmitArrays(engine, consumer, consume, R::Read(temporary), R::Write(output));
    CHECK_FALSE(ready.Ready());
    CHECK_THROWS(engine.ReleaseOn(std::move(temporary), producer));
    REQUIRE(temporary.IsValid());
    CHECK_THROWS(output.Read(consumer.Context()));
    // Enqueue returned while the producer is held. No host observation or
    // device-wide synchronization was inserted by checking or ReleaseOn.
    producer.WaitFor(consumer.Record());
    engine.ReleaseOn(std::move(temporary), producer);
    *gate.open = true;
    engine.Synchronize();
    const auto host = output.To(R::Device::Host());
    const auto read = host.Read();
    for (DNDS::index iRow = 0; iRow < 19; ++iRow)
        for (DNDS::index j = 0; j < 7; ++j)
            CHECK(read.At(iRow, j) == 3);
}

static R::Buffer Pinned(size_t bytes, std::atomic<int> &releases)
{
    void *pointer = nullptr;
    DNDS_check_throw_info(cudaMallocHost(&pointer, bytes) == cudaSuccess, "test pinned allocation failed");
    auto owner = std::shared_ptr<void>(pointer, [&releases](void *p)
                                       {
        if (cudaFreeHost(p) != cudaSuccess) std::terminate();
        ++releases; });
    return R::Buffer::FromAllocation(R::Allocation::Import(std::move(owner), bytes), 0, bytes);
}

TEST_CASE("Eager CUDA uses explicit prefix events with retained DMA owners and a three stream join")
{
    std::atomic<int> uploadedReleased{0}, downloadedReleased{0};
    R::EagerEngine engine(2);
    auto producer = std::make_shared<R::CUDALauncher>(0);
    auto reader1 = std::make_shared<R::CUDALauncher>(0);
    auto reader2 = std::make_shared<R::CUDALauncher>(0);
    auto compute = engine.CreateStream(producer);
    auto left = engine.CreateStream(reader1), right = engine.CreateStream(reader2);
    CHECK(compute.Context().stream != nullptr);
    CHECK(compute.Context().stream != left.Context().stream);
    auto shape = R::RowLayout::Dense(257, 6, 8);
    R::Array<double, R::FixedRows<6>> state(shape, R::Device::CUDA(0));
    R::Array<double> first(R::RowLayout::Ragged(std::vector<size_t>(257, 6)), R::Device::CUDA(0));
    std::vector<size_t> lengths(257);
    for (size_t i = 0; i < lengths.size(); ++i)
        lengths[i] = i % 7;
    R::Array<double> second(R::RowLayout::Padded(lengths, 9), R::Device::CUDA(0));
    auto upload = Pinned(shape.Elements() * sizeof(double), uploadedReleased);
    auto download = Pinned(first.Storage().Bytes(), downloadedReleased);
    {
        auto write = upload.Write();
        auto values = write.View<double>();
        for (size_t i = 0; i < values.size; ++i)
            values.data[i] = double(i);
    }
    PreloadKernels();
    Gate gate;
    gate.Enqueue(engine, compute);
    auto stateBytes = state.Storage();
    R::CopyAsyncOptions copyLifetime;
    copyLifetime.orderedDestination = true;
    R::CopyAsync(engine, compute, upload, stateBytes, copyLifetime);
    stateBytes = {};
    CHECK_THROWS(upload.Write());
    upload = {};
    CHECK(uploadedReleased == 0);
    for (int i = 0; i < 17; ++i)
        R::SubmitArrays(engine, compute, LaunchAdd, R::WithOrderedHandoff(R::ReadWrite(state)));
    CHECK(producer->NativeEventsRecorded() == 0);
    auto ready = compute.Record();
    CHECK(producer->NativeEventsRecorded() == 1);
    CHECK_FALSE(ready.Ready());
    CHECK_THROWS(state.Read(compute.Context()));
    left.WaitFor(ready);
    right.WaitFor(ready);
    R::SubmitArrays(engine, left, [](R::ExecutionContext context, R::ArrayView<const double> in, R::ArrayView<double> out)
                    {
        Scale<<<9, 32, 0, static_cast<cudaStream_t>(context.stream)>>>(in, out, 2);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "test Scale launch failed"); }, R::WithOrderedHandoff(R::Read(state)), R::Write(first));
    R::SubmitArrays(engine, right, [](R::ExecutionContext context, R::ArrayView<const double> in, R::ArrayView<double> out)
                    {
        Scale<<<9, 32, 0, static_cast<cudaStream_t>(context.stream)>>>(in, out, 3);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "test Scale launch failed"); }, R::WithOrderedHandoff(R::Read(state)), R::Write(second));
    left.WaitFor(right.Record()); // All independent readers precede release here.
    std::weak_ptr<const R::ArrayStorage> original = state.SharedStorage();
    engine.ReleaseOn(std::move(state), left);
    R::CopyAsync(engine, left, first.Storage(), download);
    auto joined = left.Record();
    CHECK(reader1->NativeEventsRecorded() == 1);
    CHECK(original.expired()); // CPU handles are gone; native free is queued after the joined readers.
    CHECK_THROWS(download.Read());
    auto cpu = engine.CreateCPUStream();
    std::atomic<bool> observed{false};
    cpu.WaitFor(joined);
    engine.Submit(cpu, [&observed](R::ExecutionContext)
                  { observed = true; });
    CHECK_FALSE(observed);
    *gate.open = true;
    engine.Synchronize();
    CHECK(observed);
    CHECK(original.expired());
    CHECK(uploadedReleased == 1);
    auto downloadedAccess = download.Read();
    auto result = downloadedAccess.View<double>();
    for (size_t row = 0; row < 257; ++row)
        for (size_t col = 0; col < 6; ++col)
            CHECK(result.data[row * 6 + col] == 2 * double(row * 8 + col + 17));
    auto host = second.To(R::Device::Host(), right.Context());
    auto read = host.Read();
    for (size_t row = 0; row < 257; ++row)
        for (size_t col = 0; col < lengths[row]; ++col)
            CHECK(read.At(row, col) == 3 * double(row * 8 + col + 17));
    CHECK(ready.Ready());
}

TEST_CASE("CUDA owned CSR stencil finishes while ghost writes are pending then joins ring work")
{
    constexpr size_t rows = 257, ghosts = 19, coreRows = rows - ghosts;
    std::atomic<int> releases{0};
    R::EagerEngine engine;
    auto compute = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    auto comm = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    R::Array<double> host(R::RowLayout::Dense(rows, 6, 8));
    std::vector<size_t> widths(coreRows);
    for (size_t i = 0; i < coreRows; ++i)
        widths[i] = i % 7;
    R::Array<int64_t> hostNeighbors(R::RowLayout::Ragged(widths));
    {
        auto u = host.Write();
        auto c2c = hostNeighbors.Write();
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < 6; ++j)
                u.At(i, j) = 10 * i + j;
        for (size_t i = 0; i < coreRows; ++i)
            for (size_t j = 0; j < widths[i]; ++j)
                c2c.At(i, j) = (17 * i + 13 * j) % rows;
    }
    auto owned = host.To(R::Device::CUDA(0));
    auto neighbors = hostNeighbors.To(R::Device::CUDA(0));
    R::Array<double> ghost(R::RowLayout::Dense(ghosts, 6, 8), R::Device::CUDA(0));
    R::Array<double> output(host.Layout(), R::Device::CUDA(0));
    auto downloaded = Pinned(output.Storage().Bytes(), releases);
    PreloadKernels();
    Gate gate;
    gate.Enqueue(engine, comm);
    R::SubmitArrays(engine, comm, LaunchAdd, R::Write(ghost));
    auto ghostReady = comm.Record();
    R::SubmitArrays(engine, compute, [](R::ExecutionContext context, auto c2c, auto u, auto rhs)
                    {
        CoreStencil<<<11, 32, 0, static_cast<cudaStream_t>(context.stream)>>>(c2c, u, rhs);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "test stencil launch failed"); }, R::Read(neighbors), R::Read(owned), R::Write(output));
    engine.ReleaseOn(std::move(neighbors), compute); // Device metadata follows the last stencil use.
    R::CopyAsync(engine, compute, output.Storage(), downloaded);
    auto coreReady = compute.Record();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!coreReady.Ready() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const bool completedIndependently = coreReady.Ready();
    CHECK_FALSE(ghostReady.Ready());
    CHECK_THROWS(ghost.Read(comm.Context()));
    if (!completedIndependently)
        *gate.open = true;
    REQUIRE(completedIndependently);
    {
        auto read = downloaded.Read();
        auto values = read.View<double>();
        for (size_t i = 0; i < coreRows; ++i)
            for (size_t j = 0; j < 6; ++j)
            {
                double expected = 0;
                for (size_t k = 0; k < widths[i]; ++k)
                    expected += 10 * ((17 * i + 13 * k) % rows) + j;
                CHECK(values.data[i * 8 + j] == expected);
            }
    }
    compute.WaitFor(ghostReady);
    R::SubmitArrays(engine, compute, [](R::ExecutionContext context, auto g, auto rhs)
                    {
        RingUpdate<<<3, 32, 0, static_cast<cudaStream_t>(context.stream)>>>(g, rhs);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "test ring launch failed"); }, R::Read(ghost), R::ReadWrite(output));
    *gate.open = true;
    engine.Synchronize();
    auto result = output.To(R::Device::Host());
    auto read = result.Read();
    for (size_t i = coreRows; i < rows; ++i)
        for (size_t j = 0; j < 6; ++j)
            CHECK(read.At(i, j) == 1);
}

TEST_CASE("CUDA fixed prefixes retire prior batches and preserve delayed ownership")
{
    R::EagerEngine engine;
    auto backend = std::make_shared<R::CUDALauncher>(0);
    auto stream = engine.CreateStream(backend);
    R::Array<double> array(R::RowLayout::Dense(73, 5, 8), R::Device::CUDA(0));
    R::SubmitArrays(engine, stream, LaunchAdd, R::Write(array));
    auto first = stream.Record();
    first.WaitHost();
    auto before = array.Version();
    Gate gate;
    gate.Enqueue(engine, stream);
    R::SubmitArrays(engine, stream, LaunchAdd, R::ReadWrite(array));
    auto second = stream.Record();
    CHECK(first.Ready());
    CHECK_FALSE(second.Ready());
    CHECK(array.Version() == before);
    *gate.open = true;
    second.WaitHost();
    CHECK(array.Version() == before + 1);
    CHECK(array.To(R::Device::Host(), stream.Context()).Read().At(72, 4) == 2);
}

TEST_CASE("CUDA launch exceptions drain submitted uses and invalidate partial and skipped writes")
{
    std::atomic<int> released{0};
    R::EagerEngine engine;
    auto backend = std::make_shared<R::CUDALauncher>(0);
    auto stream = engine.CreateStream(backend);
    R::Array<double> partial(R::RowLayout::Dense(131, 7, 9), R::Device::CUDA(0));
    R::Array<double> skipped(partial.Layout(), R::Device::CUDA(0));
    PreloadKernels();
    Gate gate;
    gate.Enqueue(engine, stream);
    R::SubmitArrays(engine, stream, [](R::ExecutionContext context, R::ArrayView<double> values)
                    {
        LaunchAdd(context, values);
        throw std::runtime_error("host failure after launch"); }, R::Write(partial));
    bool launched = false;
    R::SubmitArrays(engine, stream, [&launched](R::ExecutionContext context, R::ArrayView<double> values)
                    {
        launched = true;
        LaunchAdd(context, values); }, R::Write(skipped));
    auto owner = std::shared_ptr<int>(new int(3), [&](int *p)
                                      { delete p; ++released; });
    engine.ReleaseOn(std::move(owner), stream);
    auto tail = stream.Record();
    CHECK_FALSE(launched);
    CHECK_FALSE(tail.Ready());
    CHECK(released == 0);
    *gate.open = true;
    CHECK_THROWS(engine.Synchronize());
    CHECK_FALSE(partial.IsValid());
    CHECK_FALSE(skipped.IsValid());
    CHECK(partial.Storage().IsIdle());
    CHECK(released == 1);
    // A retired failure remains observable without repeating retirement or
    // dropping its terminal error on the completed-prefix fast path.
    for (int repeat = 0; repeat < 4; ++repeat)
    {
        CHECK_THROWS(tail.Ready());
        CHECK_THROWS(tail.WaitHost());
    }
    CHECK(released == 1);
}

TEST_CASE("Concurrent CUDA host submitters preserve one native stream order without kernel events")
{
    R::EagerEngine engine;
    auto backend = std::make_shared<R::CUDALauncher>(0);
    auto stream = engine.CreateStream(backend);
    R::Array<double> state(R::RowLayout::Dense(43, 6, 8), R::Device::CUDA(0));
    std::vector<std::thread> threads;
    std::atomic<bool> failed{false};
    for (int i = 0; i < 4; ++i)
        threads.emplace_back([&]
                             {
            try { for (int j = 0; j < 17; ++j) R::SubmitArrays(engine, stream, LaunchAdd, R::ReadWrite(state)); }
            catch (...) { failed = true; } });
    for (auto &thread : threads)
        thread.join();
    CHECK_FALSE(failed);
    CHECK(backend->NativeEventsRecorded() == 0);
    stream.Record().WaitHost();
    CHECK(backend->NativeEventsRecorded() == 1);
    CHECK(state.Version() == 68);
    CHECK(state.To(R::Device::Host(), stream.Context()).Read().At(42, 5) == 68);
}

TEST_CASE("CUDA dependency only prefixes retire earlier batches and completed history is discarded")
{
    R::StreamEvent survivor;
    R::Array<double> a(R::RowLayout::Dense(53, 6, 8), R::Device::CUDA(0));
    R::Array<double> b(a.Layout(), R::Device::CUDA(0));
    {
        R::EagerEngine engine;
        auto first = std::make_shared<R::CUDALauncher>(0);
        auto second = std::make_shared<R::CUDALauncher>(0);
        auto left = engine.CreateStream(first), right = engine.CreateStream(second);
        PreloadKernels();
        Gate gate;
        gate.Enqueue(engine, left);
        R::SubmitArrays(engine, left, LaunchAdd, R::Write(a));
        auto earlier = left.Record();
        std::weak_ptr<const R::StreamEvent::State> old = earlier.Provider();
        R::SubmitArrays(engine, right, LaunchAdd, R::Write(b));
        left.WaitFor(right.Record());
        auto join = left.Record(); // Contains a wait, no kernel.
        CHECK_FALSE(join.Ready());
        *gate.open = true;
        join.WaitHost();
        CHECK(a.Storage().IsIdle());
        CHECK(b.Storage().IsIdle());
        CHECK(a.Version() == 1);
        earlier = {};
        join = {};
        for (int iteration = 0; iteration < 97; ++iteration)
        {
            R::SubmitArrays(engine, left, LaunchAdd, R::ReadWrite(a));
            left.Record().WaitHost();
        }
        CHECK(old.expired());
        CHECK(first->NativeEventsRecorded() == 99);
        survivor = left.Record(); // Same immutable prefix; no extra event.
        CHECK(first->NativeEventsRecorded() == 99);
        struct Unready : R::StreamEvent::State
        {
            bool Ready() const override { return false; }
            void Wait() const override { throw std::runtime_error("must not host-wait implicitly"); }
        };
        CHECK_THROWS(left.WaitFor(R::StreamEvent(std::make_shared<Unready>())));
        engine.Synchronize();
    }
    // Completed observations outlive engine/stream teardown and do not rerun
    // captured retirement. The provider caches both completion and its error.
    for (int repeat = 0; repeat < 16; ++repeat)
    {
        CHECK(survivor.Ready());
        CHECK_NOTHROW(survivor.WaitHost());
    }
    CHECK(a.To(R::Device::Host()).Read().At(52, 5) == 98);
}

TEST_CASE("CUDA async allocation supports dynamic tables and prepared padded CSR metadata behind pending work")
{
    R::EagerEngine engine;
    auto producer = std::make_shared<R::CUDALauncher>(0), consumer = std::make_shared<R::CUDALauncher>(0);
    auto first = engine.CreateStream(producer), last = engine.CreateStream(consumer);
    std::vector<size_t> lengths(193);
    for (size_t i = 0; i < lengths.size(); ++i)
        lengths[i] = i % 9;
    R::Array<double> paddedPrototype(R::RowLayout::Padded(lengths, 11), R::Device::CUDA(0));
    R::Array<double> csrPrototype(R::RowLayout::Ragged(lengths), R::Device::CUDA(0));
    R::Array<double> result(R::RowLayout::Dense(257, 6, 8), R::Device::CUDA(0));
    PreloadKernels();
    Gate gate;
    gate.Enqueue(engine, first);
    auto table = R::AllocateArrayAsync<double, R::FixedRows<6>>(engine, first, result.Layout());
    auto padded = R::AllocateLikeAsync(engine, first, paddedPrototype);
    auto csr = R::AllocateLikeAsync(engine, first, csrPrototype);
    CHECK(padded.SharedStorage()->SharedLayout() == paddedPrototype.SharedStorage()->SharedLayout());
    CHECK(csr.SharedStorage()->SharedLayout() == csrPrototype.SharedStorage()->SharedLayout());
    CHECK_THROWS(table.Read(first.Context()));
    CHECK_THROWS(R::AllocateArrayAsync<double>(engine, first, R::RowLayout::Ragged({1, 3})));
    CHECK_THROWS((R::AllocateArrayAsync<double, R::FixedRows<5>>(engine, first, result.Layout())));
    auto alias = table;
    for (int i = 0; i < 7; ++i)
    {
        R::SubmitArrays(engine, first, LaunchAdd, R::WithOrderedHandoff(R::ReadWrite(table)));
        R::SubmitArrays(engine, first, LaunchAdd, R::ReadWrite(padded));
        R::SubmitArrays(engine, first, LaunchAdd, R::ReadWrite(csr));
    }
    engine.ReleaseOn(std::move(table), first); // Non-final: alias still owns values.
    last.WaitFor(first.Record());
    R::SubmitArrays(engine, last, [](R::ExecutionContext context, R::ArrayView<const double> in, R::ArrayView<double> out)
                    {
        Scale<<<9, 32, 0, static_cast<cudaStream_t>(context.stream)>>>(in, out, 3);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "test scale failed"); }, R::WithOrderedHandoff(R::Read(alias)), R::Write(result));
    auto values = std::weak_ptr<const R::ArrayStorage>(alias.SharedStorage());
    engine.ReleaseOn(std::move(alias), last); // Caller alias released; initialization may still retain the allocation.
    CHECK(values.expired());
    auto empty = engine.AllocateBuffer(first, 0);
    engine.ReleaseOn(std::move(empty), first);
    CHECK(producer->NativeEventsRecorded() == 1);
    CHECK(consumer->NativeEventsRecorded() == 0);
    *gate.open = true;
    engine.Synchronize();
    CHECK(result.To(R::Device::Host(), last.Context()).Read().At(256, 5) == 21);
    auto hostPadded = padded.To(R::Device::Host(), first.Context());
    auto hostCSR = csr.To(R::Device::Host(), first.Context());
    auto p = hostPadded.Read(), c = hostCSR.Read();
    for (size_t row = 0; row < lengths.size(); ++row)
        for (size_t col = 0; col < lengths[row]; ++col)
        {
            CHECK(p.At(row, col) == 7);
            CHECK(c.At(row, col) == 7);
        }
    CHECK(padded.Version() == 8); // Async zero initialization plus seven updates.
    engine.ReleaseOn(std::move(padded), last);
    engine.ReleaseOn(std::move(csr), last);
    engine.Synchronize();
}

TEST_CASE("Backend allocation resources receive the final selected free stream across independent handles")
{
    struct Observation
    {
        int releases = 0;
        void *stream = nullptr;
    } observed;
    struct Resource : R::AllocationResource
    {
        void *pointer = nullptr;
        Observation &observed;
        explicit Resource(Observation &o) : observed(o)
        {
            DNDS_check_throw_info(cudaMalloc(&pointer, Bytes()) == cudaSuccess, "test allocation failed");
        }
        ~Resource() override { Release(nullptr); }
        void *Data() const noexcept override { return pointer; }
        size_t Bytes() const noexcept override { return 31 * 7 * sizeof(double); }
        R::Device GetDevice() const noexcept override { return R::Device::CUDA(0); }
        bool CanReleaseOrdered(R::ExecutionContext c) const noexcept override { return c.device == GetDevice(); }
        void Release(const R::ExecutionContext *c) noexcept override
        {
            if (!pointer)
                return;
            observed.stream = c ? c->stream : nullptr;
            auto status = c ? cudaFreeAsync(pointer, static_cast<cudaStream_t>(c->stream)) : cudaFree(pointer);
            if (status != cudaSuccess)
                std::terminate();
            pointer = nullptr;
            ++observed.releases;
        }
    };
    R::EagerEngine engine;
    auto a = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    auto b = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    auto buffer = R::Buffer::FromResource(std::make_unique<Resource>(observed));
    auto array = R::Array<double>::FromStorage(R::RowLayout::Dense(31, 5, 7), std::move(buffer));
    auto alias = array;
    PreloadKernels();
    Gate gate;
    gate.Enqueue(engine, a);
    R::SubmitArrays(engine, a, [](R::ExecutionContext c, R::ArrayView<double> u)
                    {
        DNDS_check_throw_info(cudaMemsetAsync(u.data, 0, u.RowCount() * u.stride * sizeof(double), static_cast<cudaStream_t>(c.stream)) == cudaSuccess,
                              "test initialization failed");
        LaunchAdd(c, u); }, R::WithOrderedHandoff(R::Write(array)));
    engine.ReleaseOn(std::move(array), a);
    CHECK(observed.releases == 0);
    b.WaitFor(a.Record());
    R::SubmitArrays(engine, b, LaunchAdd, R::WithOrderedHandoff(R::ReadWrite(alias)));
    engine.ReleaseOn(std::move(alias), b);
    CHECK(observed.releases == 1);
    CHECK(observed.stream == b.Context().stream);
    *gate.open = true;
    engine.Synchronize();
    CHECK(observed.releases == 1);
}

TEST_CASE("CUDA aligned physical tables cross streams and release after the final use")
{
    for (const auto &shape : {R::RowLayout::ColMajor(33, 21, {32, 1}), R::RowLayout::TiledColMajor(33, 21, 8)})
    {
        R::EagerEngine engine;
        auto first = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
        auto last = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
        auto temporary = R::AllocateArrayAsync<double>(engine, first, shape, R::AllocationOptions{4096});
        CHECK(reinterpret_cast<uintptr_t>(temporary.Storage().LegacyPointer()) % 4096 == 0);
        auto alias = temporary;
        R::Array<double> result(R::RowLayout::Dense(33, 21), R::Device::CUDA(0));
        for (int step = 0; step < 3; ++step)
            R::SubmitArrays(engine, first, LaunchAdd, R::ReadWrite(temporary));
        engine.ReleaseOn(std::move(temporary), first);
        last.WaitFor(first.Record());
        R::SubmitArrays(engine, last, [](R::ExecutionContext context, auto input, auto output)
                        {
            Scale<<<7, 32, 0, static_cast<cudaStream_t>(context.stream)>>>(input, output, 7);
            DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "aligned table scale failed"); }, R::Read(alias), R::Write(result));
        engine.ReleaseOn(std::move(alias), last);
        engine.Synchronize();
        const auto host = result.To(R::Device::Host());
        const auto read = host.Read();
        for (DNDS::index iRow = 0; iRow < 33; ++iRow)
            for (DNDS::index iComp = 0; iComp < 21; ++iComp)
                CHECK(read.View()(iRow, iComp) == 21);
    }
}
