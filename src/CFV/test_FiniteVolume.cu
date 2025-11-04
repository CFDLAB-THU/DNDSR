#define DNDS_NDEBUG_DEVICE
#include "test_FiniteVolume.hpp"
#include <thrust/device_ptr.h>
#include <thrust/device_malloc_allocator.h>
#include "DNDS/CUDA_Utils.hpp"

namespace DNDS::CFV
{
    static const DeviceBackend B = DeviceBackend::CUDA;
    DNDS_GLOBAL void finiteVolumeCellOpTest_run_CUDA_kernel(
        FiniteVolume::t_deviceView<B> *fv,
        tUDof<DynamicSize>::t_deviceView<B> *u,
        tUGrad<DynamicSize, 3>::t_deviceView<B> *u_grad)
    {
        // static const size_t formal_param_size = sizeof(fv) + sizeof(u) + sizeof(u_grad);
        index tid = blockIdx.x * blockDim.x + threadIdx.x;
        index iCell = tid;
        if (iCell >= fv->mesh.NumCell())
            return;
        finiteVolumeCellOpTest(*fv, *u, *u_grad, iCell);
    }

    template <>
    void finiteVolumeCellOpTest_run<B>(
        FiniteVolume::t_deviceView<B> &fv,
        tUDof<DynamicSize>::t_deviceView<B> &u,
        tUGrad<DynamicSize, 3>::t_deviceView<B> &u_grad)
    {
        index threadsPerBlock = 512;
        index blocksPerGrid = (fv.mesh.NumCell() + threadsPerBlock - 1) / threadsPerBlock;

        finiteVolumeCellOpTest_run_CUDA_kernel<<<threadsPerBlock, blocksPerGrid>>>(
            DNDS_CUDA_DEVICE_VIEW_TMP_COPY(fv),
            DNDS_CUDA_DEVICE_VIEW_TMP_COPY(u),
            DNDS_CUDA_DEVICE_VIEW_TMP_COPY(u_grad));
    }
}