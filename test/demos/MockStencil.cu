#include "MockStencil.hpp"
#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include <cuda_runtime.h>

namespace DNDS::Runtime::Demo
{
    __global__ void Core(ArrayView<const double> u, ArrayView<double> v)
    {
        for (size_t i = 1 + blockIdx.x * blockDim.x + threadIdx.x; i + 1 < u.Rows(); i += blockDim.x * gridDim.x)
            Cell(u, {}, v, i);
    }
    __global__ void Ring(ArrayView<const double> u, ArrayView<const double> g, ArrayView<double> v)
    {
        Cell(u, g, v, 0);
        Cell(u, g, v, u.Rows() - 1);
    }
    void StencilCUDA(EagerEngine &engine, const ExecutionStream &stream, State owned, State ghost, State output, bool core)
    {
        if (core)
            SubmitArrays(engine, stream, [](ExecutionContext c, ArrayView<const double> u, ArrayView<double> v)
                         {
                Core<<<17, 128, 0, CUDAStream(c)>>>(u, v);
                DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "mock core launch failed"); }, Read(owned), Write(output));
        else
            SubmitArrays(engine, stream, [](ExecutionContext c, ArrayView<const double> u, ArrayView<const double> g, ArrayView<double> v)
                         {
                Ring<<<1, 1, 0, CUDAStream(c)>>>(u, g, v);
                DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "mock ring launch failed"); }, Read(owned), Read(ghost), ReadWrite(output));
    }
}
