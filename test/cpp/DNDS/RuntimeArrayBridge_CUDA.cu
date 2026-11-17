#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <thread>
#include <chrono>

namespace R = DNDS::Runtime;
using State = R::Array<double, R::FixedRows<5>>;

__global__ void ScaleSharedState(R::ArrayView<double> values, double factor)
{
    for (size_t i = blockIdx.x; i < values.RowCount(); i += gridDim.x)
        for (size_t j = threadIdx.x; j < values.RowSize(i); j += blockDim.x)
            values(i, j) *= factor;
}

void BridgeScaleCUDA(State &array, double factor)
{
    auto context = R::ExecutionContext{array.GetDevice(), nullptr};
    auto access = array.Write(context);
    ScaleSharedState<<<3, 32>>>(access.View(), factor);
    DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "bridge kernel launch failed");
    access.Complete(R::Completion::Record(context));
    access.Wait();
}

void BridgeScaleCUDAAsync(R::EagerEngine &engine, const R::ExecutionStream &stream, State &array, double factor)
{
    R::SubmitArrays(engine, stream, [factor](R::ExecutionContext context, R::ArrayView<double> values)
                    {
        ScaleSharedState<<<7, 32, 0, ::DNDS::Runtime::CUDAStream(context)>>>(values, factor);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "bridge async scale failed"); }, R::ReadWrite(array));
}

void BridgeHoldCUDA(R::EagerEngine &engine, const R::ExecutionStream &stream, std::shared_ptr<std::atomic<bool>> flag)
{
    cudaFuncAttributes attributes{};
    DNDS_check_throw_info(cudaFuncGetAttributes(&attributes, ScaleSharedState) == cudaSuccess, "bridge preload failed");
    engine.Submit(stream, [flag = std::move(flag)](R::ExecutionContext context)
                  {
        auto status = cudaLaunchHostFunc(::DNDS::Runtime::CUDAStream(context), [](void *pointer)
        {
            auto *open = static_cast<std::atomic<bool> *>(pointer);
            while (!open->load()) std::this_thread::sleep_for(std::chrono::microseconds(50));
        }, flag.get());
        DNDS_check_throw_info(status == cudaSuccess, "bridge gate failed"); });
}
