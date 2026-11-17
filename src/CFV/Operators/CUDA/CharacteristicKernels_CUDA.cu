#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "CFV/Detail/CharacteristicKernels.hpp"
#include <cuda_runtime.h>
#include <cmath>

namespace DNDS::CFV::Detail
{
    namespace R = Runtime;
    namespace
    {
        __global__ void ValidateCharacteristics(DeviceCharacteristicView transform, R::ArrayView<int> status)
        {
            for (index iFace = index(blockIdx.x) * blockDim.x + threadIdx.x; iFace < transform.forward.rows; iFace += index(gridDim.x) * blockDim.x)
            {
                bool valid = true;
                const index n = transform.count;
                for (index j = 0; j < n * n; ++j)
                    valid &= isfinite(transform.forward(iFace, j)) && isfinite(transform.inverse(iFace, j));
                real errorLR = 0, errorRL = 0, normLR = 0, normRL = 0;
                for (index i = 0; i < n; ++i)
                    for (index j = 0; j < n; ++j)
                    {
                        real lr = 0, rl = 0;
                        for (index k = 0; k < n; ++k)
                        {
                            lr += transform.forward(iFace, i + n * k) * transform.inverse(iFace, k + n * j);
                            rl += transform.inverse(iFace, i + n * k) * transform.forward(iFace, k + n * j);
                        }
                        normLR += lr * lr;
                        normRL += rl * rl;
                        lr -= i == j;
                        rl -= i == j;
                        errorLR += lr * lr;
                        errorRL += rl * rl;
                    }
                // Eigen isApprox-style relative Frobenius criterion, matching
                // host finalization tolerance and checking both directions.
                valid &= isfinite(errorLR) && isfinite(errorRL) && isfinite(normLR) && isfinite(normRL) &&
                         errorLR <= 1e-20 * fmin(normLR, real(n)) && errorRL <= 1e-20 * fmin(normRL, real(n));
                if (!valid)
                    atomicExch(&status(0, 0), 1);
            }
        }
    }
    void LaunchValidateCharacteristics(R::ExecutionContext context, DeviceCharacteristicView transform, R::ArrayView<int> status)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "queued characteristic validation requires CUDA");
        if (!transform.count)
            return;
        R::ValidateViewDevices(context.device, transform, status);
        DNDS_check_throw_info(transform.count > 0 && transform.count <= 64 && transform.first >= 0 &&
                                  transform.forward.rows == transform.inverse.rows && transform.forward.width == transform.count * transform.count &&
                                  transform.inverse.width == transform.forward.width && status.rows == 1 && status.width == 1,
                              "queued characteristic validation shape mismatch");
        R::CheckKernelArgumentBudget<DeviceCharacteristicView, R::ArrayView<int>>();
        if (!transform.forward.rows)
            return;
        ValidateCharacteristics<<<unsigned(std::min<index>(256, (transform.forward.rows + 127) / 128)), 128, 0,
                                  ::DNDS::Runtime::CUDAStream(context)>>>(transform, status);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "queued characteristic validation launch failed");
    }
}
