#pragma once

#include "DNDS/Runtime/Storage/Buffer.hpp"
#ifdef DNDS_USE_CUDA
#    include <cuda_runtime_api.h>

namespace DNDS::Runtime
{
    /// Backend boundary only. Rejects a foreign/retired resource before launch;
    /// the enclosing context retains the resource through the native call.
    cudaStream_t CUDAStream(const ExecutionContext &context);
    cudaStream_t CUDAStream(const StreamResource &resource);
    std::shared_ptr<StreamResource> CreateCUDAStream(int ordinal);
    /// Transfer ownership of a native CUDA stream to the abstract resource.
    std::shared_ptr<StreamResource> AdoptCUDAStream(int ordinal, cudaStream_t stream);
    /// Foreign backend adapter: owner must retain the actual external stream.
    std::shared_ptr<StreamResource> ImportCUDAStream(int ordinal, cudaStream_t stream, std::shared_ptr<const void> owner);
    /// Explicit foreign-ABI loan (DLPack/legacy only). That ABI supplies no owner;
    /// its caller must keep the native stream alive through the exported lease.
    /// Native solver/orchestration paths must use owned/imported resources.
    std::shared_ptr<StreamResource> BorrowExternalCUDAStream(int ordinal, cudaStream_t stream);
}
#endif
