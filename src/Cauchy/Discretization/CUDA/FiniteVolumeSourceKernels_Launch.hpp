#pragma once
#include "Cauchy/Discretization/CUDA/FiniteVolumeSourceKernels.hpp"
#include <cuda_runtime.h>
namespace DNDS::Cauchy::CUDA::Detail
{
    enum class VolumeSourceModel
    {
        ACM,
        PerfectGas
    };
    /// M=0 consumes a physical gradient. M>0 samples the finalized modal row.
    template <VolumeSourceModel Model, int C, int D, int M>
    void LaunchVolumeSourceCells(cudaStream_t, SourceRequest, Runtime::RowSelection, unsigned blocks);
}
