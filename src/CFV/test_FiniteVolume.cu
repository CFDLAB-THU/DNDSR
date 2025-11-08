#define DNDS_NDEBUG_DEVICE
#include "test_FiniteVolume.hpp"
#include <thrust/device_ptr.h>
#include <thrust/device_malloc_allocator.h>
#include "DNDS/CUDA_Utils.hpp"

namespace DNDS::CFV
{
    static const DeviceBackend B = DeviceBackend::CUDA;
    static const auto size_of_fv_view = sizeof(Geom::tAdjPair::t_deviceView<DeviceBackend::CUDA>);

    
    DNDS_GLOBAL void finiteVolumeCellOpTest_run_CUDA_kernel(
        FiniteVolume::t_deviceView<B> *fv,
        tUDof<DynamicSize>::t_deviceView<B> *u,
        tUGrad<DynamicSize, 3>::t_deviceView<B> *u_grad)
    {
        index tid = blockIdx.x * blockDim.x + threadIdx.x;
        index iCell = tid;
        // if (iCell == 0)
        //     printf("fuck, %ld\n", fv->mesh.NumCell());
        if (iCell >= fv->mesh.NumCell())
            return;
        finiteVolumeCellOpTest(*fv, *u, *u_grad, iCell);
    }

    template <>
    void finiteVolumeCellOpTest_run<B>(
        FiniteVolume::t_deviceView<B> &fv,
        tUDof<DynamicSize>::t_deviceView<B> &u,
        tUGrad<DynamicSize, 3>::t_deviceView<B> &u_grad,
        const t_jsonconfig &settings)
    {
        index threadsPerBlock = 32;
        if (settings.contains("threadsPerBlock"))
            threadsPerBlock = settings["threadsPerBlock"];

        index blocksPerGrid = (fv.mesh.NumCell() + threadsPerBlock - 1) / threadsPerBlock;
        // std::cout << blocksPerGrid << " ,, " << threadsPerBlock << std::endl;
        // 
        // auto fv_d = CUDA::DeviceObject<FiniteVolume::t_deviceView<B>>(fv);
        finiteVolumeCellOpTest_run_CUDA_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            DNDS_CUDA_DEVICE_VIEW_TMP_COPY(fv),
            DNDS_CUDA_DEVICE_VIEW_TMP_COPY(u),
            DNDS_CUDA_DEVICE_VIEW_TMP_COPY(u_grad));
        cudaDeviceSynchronize();
    }
}