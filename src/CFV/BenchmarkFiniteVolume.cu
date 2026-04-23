#define DNDS_NDEBUG_DEVICE
#include "BenchmarkFiniteVolume.hpp"
#include <thrust/device_ptr.h>
#include <thrust/device_malloc_allocator.h>
#include "DNDS/Device/CUDA_Utils.hpp"

namespace DNDS::CFV
{
    static const DeviceBackend B = DeviceBackend::CUDA;
    static const auto size_of_fv_view = sizeof(Geom::tAdjPair::t_deviceView<DeviceBackend::CUDA>);
    auto a = std::is_trivially_copyable_v<DNDS::CFV::FiniteVolumeDeviceView<DNDS::DeviceBackend::CUDA>>;
    DNDS_GLOBAL void finiteVolumeCellOpTest_run_CUDA_kernel(
        FiniteVolume::t_deviceView<B> *fv,
        tUDof<DynamicSize>::t_deviceView<B> *u,
        tUGrad<DynamicSize, 3>::t_deviceView<B> *u_grad)
    {
        index tid = blockIdx.x * blockDim.x + threadIdx.x;
        index iCell = tid;
        if (iCell >= fv->mesh.NumCell())
            return;
        uint32_t nVars = u->father.MatRowSize();
        extern __shared__ real sdata[];
        finiteVolumeCellOpTest(*fv, *u, *u_grad, iCell, sdata + threadIdx.x * (nVars * 3));
    }

    DNDS_GLOBAL void finiteVolumeCellOpTest_run_CUDA_kernel_var_per_thread(
        FiniteVolume::t_deviceView<B> *fv,
        tUDof<DynamicSize>::t_deviceView<B> *u,
        tUGrad<DynamicSize, 3>::t_deviceView<B> *u_grad,
        int cells_per_block,
        int group_size)
    {
        int tid = threadIdx.x;
        index iCell = cells_per_block * blockIdx.x + tid / group_size;
        int iVar = tid % group_size;

        if (iCell >= fv->mesh.NumCell())
            return;
        finiteVolumeCellOpTest<B, true>(*fv, *u, *u_grad, iCell, nullptr, iVar);
    }

    template <>
    void finiteVolumeCellOpTest_run<B>(
        FiniteVolume::t_deviceView<B> &fv,
        tUDof<DynamicSize>::t_deviceView<B> &u,
        tUGrad<DynamicSize, 3>::t_deviceView<B> &u_grad,
        int nIter,
        const t_jsonconfig &settings)
    {
        int nVars = u.father.MatRowSize();
        index threadsPerBlock0 = 32;
        if (settings.contains("threadsPerBlock"))
            threadsPerBlock0 = settings["threadsPerBlock"];

        std::string method;
        if (settings.contains("method"))
            method = settings["method"];

        if (method.empty() || method == "percell")
        {
            index threadsPerBlock = threadsPerBlock0;
            index blocksPerGrid = (fv.mesh.NumCell() + threadsPerBlock - 1) / threadsPerBlock;
            // std::cout << blocksPerGrid << " ,, " << threadsPerBlock << std::endl;
            //
            // auto fv_d = CUDA::DeviceObject<FiniteVolume::t_deviceView<B>>(fv);
            int sharedMemSize = threadsPerBlock * sizeof(real) * (nVars * 3);
            for (int i = 0; i < nIter; i++)
                finiteVolumeCellOpTest_run_CUDA_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
                    DNDS_CUDA_DEVICE_VIEW_TMP_COPY(fv),
                    DNDS_CUDA_DEVICE_VIEW_TMP_COPY(u),
                    DNDS_CUDA_DEVICE_VIEW_TMP_COPY(u_grad));
        }
        else if (method == "pervar")
        {
            int v_size = u.father.MatRowSize();
            int group_size_blk = std::min(32, v_size);

            int group_size = (v_size + group_size_blk - 1) / group_size_blk * group_size_blk;

            index total_threads = group_size * fv.mesh.NumCell();
            index cells_per_blk = (threadsPerBlock0 + group_size - 1) / group_size;

            index threadsPerBlock = group_size * cells_per_blk;
            index blocksPerGrid = (total_threads + threadsPerBlock - 1) / threadsPerBlock;

            // std::cout << "v_size " << v_size << std::endl;
            // std::cout << "group_size " << group_size << std::endl;
            // std::cout << "threadsPerBlock " << threadsPerBlock << std::endl;
            // std::cout << "blocksPerGrid " << blocksPerGrid << std::endl;

            for (int i = 0; i < nIter; i++)
                finiteVolumeCellOpTest_run_CUDA_kernel_var_per_thread<<<blocksPerGrid, threadsPerBlock>>>(
                    DNDS_CUDA_DEVICE_VIEW_TMP_COPY(fv),
                    DNDS_CUDA_DEVICE_VIEW_TMP_COPY(u),
                    DNDS_CUDA_DEVICE_VIEW_TMP_COPY(u_grad),
                    cells_per_blk, group_size);
        }
        else
            DNDS_assert_info(false, "no such method: " + method);

        cudaDeviceSynchronize();
    }

    /*************************************************************/

    /*************************************************************/

    /*************************************************************/

    /*************************************************************/

    template <int nVarsFixed>
    DNDS_GLOBAL void finiteVolumeCellOpTest_Fixed_run_CUDA_kernel(
        FiniteVolume::t_deviceView<B> *fv,
        typename tUDof<nVarsFixed>::template t_deviceView<B> *u,
        typename tUGrad<nVarsFixed, 3>::template t_deviceView<B> *u_grad)
    {
        index tid = blockIdx.x * blockDim.x + threadIdx.x;
        index iCell = tid;
        if (iCell >= fv->mesh.NumCell())
            return;
        uint32_t nVars = u->father.MatRowSize();

        finiteVolumeCellOpTest_Fixed<B, nVarsFixed>(*fv, *u, *u_grad, iCell);
    }

    template <int nVarsFixed>
    DNDS_GLOBAL void finiteVolumeCellOpTest_Fixed_run_CUDA_kernel_var_per_thread(
        FiniteVolume::t_deviceView<B> *fv,
        typename tUDof<nVarsFixed>::template t_deviceView<B> *u,
        typename tUGrad<nVarsFixed, 3>::template t_deviceView<B> *u_grad,
        int cells_per_block,
        int group_size)
    {
        int tid = threadIdx.x;
        index iCell = cells_per_block * blockIdx.x + tid / group_size;
        int iVar = tid % group_size;

        if (iCell >= fv->mesh.NumCell())
            return;
        finiteVolumeCellOpTest_Fixed<B, nVarsFixed, true>(*fv, *u, *u_grad, iCell, iVar);
    }

    template <int nVarsFixed>
    void finiteVolumeCellOpTest_Fixed_entry<B, nVarsFixed>::run(
        FiniteVolume::t_deviceView<B> &fv,
        typename tUDof<nVarsFixed>::template t_deviceView<B> &u,
        typename tUGrad<nVarsFixed, 3>::template t_deviceView<B> &u_grad,
        int nIter,
        const t_jsonconfig &settings)
    {
        int nVars = nVarsFixed;
        index threadsPerBlock0 = 32;
        if (settings.contains("threadsPerBlock"))
            threadsPerBlock0 = settings["threadsPerBlock"];

        std::string method;
        if (settings.contains("method"))
            method = settings["method"];

        if (method.empty() || method == "percell")
        {
            index threadsPerBlock = threadsPerBlock0;
            index blocksPerGrid = (fv.mesh.NumCell() + threadsPerBlock - 1) / threadsPerBlock;
            // std::cout << blocksPerGrid << " ,, " << threadsPerBlock << std::endl;
            //
            // auto fv_d = CUDA::DeviceObject<FiniteVolume::t_deviceView<B>>(fv);
            // int sharedMemSize = threadsPerBlock * sizeof(real) * (nVars * 3); // no shared mem
            for (int i = 0; i < nIter; i++)
                finiteVolumeCellOpTest_Fixed_run_CUDA_kernel<nVarsFixed><<<blocksPerGrid, threadsPerBlock>>>(
                    DNDS_CUDA_DEVICE_VIEW_TMP_COPY(fv),
                    DNDS_CUDA_DEVICE_VIEW_TMP_COPY(u),
                    DNDS_CUDA_DEVICE_VIEW_TMP_COPY(u_grad));
        }
        else if (method == "pervar")
        {
            int v_size = u.father.MatRowSize();
            int group_size_blk = std::min(32, v_size);

            int group_size = (v_size + group_size_blk - 1) / group_size_blk * group_size_blk;

            index total_threads = group_size * fv.mesh.NumCell();
            index cells_per_blk = (threadsPerBlock0 + group_size - 1) / group_size;

            index threadsPerBlock = group_size * cells_per_blk;
            index blocksPerGrid = (total_threads + threadsPerBlock - 1) / threadsPerBlock;

            // std::cout << "v_size " << v_size << std::endl;
            // std::cout << "group_size " << group_size << std::endl;
            // std::cout << "threadsPerBlock " << threadsPerBlock << std::endl;
            // std::cout << "blocksPerGrid " << blocksPerGrid << std::endl;
            for (int i = 0; i < nIter; i++)
                finiteVolumeCellOpTest_Fixed_run_CUDA_kernel_var_per_thread<nVarsFixed><<<blocksPerGrid, threadsPerBlock>>>(
                    DNDS_CUDA_DEVICE_VIEW_TMP_COPY(fv),
                    DNDS_CUDA_DEVICE_VIEW_TMP_COPY(u),
                    DNDS_CUDA_DEVICE_VIEW_TMP_COPY(u_grad),
                    cells_per_blk, group_size);
        }
        else
            DNDS_assert_info(false, "no such method: " + method);

        cudaDeviceSynchronize();
    }

    template void finiteVolumeCellOpTest_Fixed_entry<B, 1>::run(
        FiniteVolume::t_deviceView<B> &fv,
        typename tUDof<1>::template t_deviceView<B> &u,
        typename tUGrad<1, 3>::template t_deviceView<B> &u_grad,
        int nIter,
        const t_jsonconfig &settings);

    template void finiteVolumeCellOpTest_Fixed_entry<B, 5>::run(
        FiniteVolume::t_deviceView<B> &fv,
        typename tUDof<5>::template t_deviceView<B> &u,
        typename tUGrad<5, 3>::template t_deviceView<B> &u_grad,
        int nIter,
        const t_jsonconfig &settings);

    /*************************************************************/

    /*************************************************************/

    /*************************************************************/

    /*************************************************************/

    template <int nVarsFixed>
    DNDS_GLOBAL void finiteVolumeCellOpTest_SOA_ver0_run_CUDA_kernel(
        FiniteVolume::t_deviceView<B> *fv,
        std::array<tUDof<1>::t_deviceView<B>, nVarsFixed> *u,
        std::array<tUGrad<1, 3>::t_deviceView<B>, nVarsFixed> *u_grad)
    {
        index tid = blockIdx.x * blockDim.x + threadIdx.x;
        index iCell = tid;
        if (iCell >= fv->mesh.NumCell())
            return;
        uint32_t nVars = nVarsFixed;

        finiteVolumeCellOpTest_SOA_ver0<B, nVarsFixed>(*fv, *u, *u_grad, iCell);
    }

    template <int nVarsFixed>
    DNDS_GLOBAL void finiteVolumeCellOpTest_SOA_ver0_run_CUDA_kernel_var_per_thread(
        FiniteVolume::t_deviceView<B> *fv,
        std::array<tUDof<1>::t_deviceView<B>, nVarsFixed> *u,
        std::array<tUGrad<1, 3>::t_deviceView<B>, nVarsFixed> *u_grad,
        int cells_per_block,
        int group_size)
    {
        int tid = threadIdx.x;
        index iCell = cells_per_block * blockIdx.x + tid / group_size;
        int iVar = tid % group_size;

        if (iCell >= fv->mesh.NumCell())
            return;
        finiteVolumeCellOpTest_SOA_ver0<B, nVarsFixed, true>(*fv, *u, *u_grad, iCell, iVar);
    }

    template <int nVarsFixed>
    void finiteVolumeCellOpTest_SOA_ver0_entry<B, nVarsFixed>::run(
        FiniteVolume::t_deviceView<B> &fv,
        std::array<tUDof<1>::t_deviceView<B>, nVarsFixed> &u,
        std::array<tUGrad<1, 3>::t_deviceView<B>, nVarsFixed> &u_grad,
        int nIter,
        const t_jsonconfig &settings)
    {
        int nVars = nVarsFixed;
        index threadsPerBlock0 = 32;
        if (settings.contains("threadsPerBlock"))
            threadsPerBlock0 = settings["threadsPerBlock"];

        std::string method;
        if (settings.contains("method"))
            method = settings["method"];

        if (method.empty() || method == "percell")
        {
            index threadsPerBlock = threadsPerBlock0;
            index blocksPerGrid = (fv.mesh.NumCell() + threadsPerBlock - 1) / threadsPerBlock;
            // std::cout << blocksPerGrid << " ,, " << threadsPerBlock << std::endl;
            //
            // auto fv_d = CUDA::DeviceObject<FiniteVolume::t_deviceView<B>>(fv);
            // int sharedMemSize = threadsPerBlock * sizeof(real) * (nVars * 3); // no shared mem
            for (int i = 0; i < nIter; i++)
                finiteVolumeCellOpTest_SOA_ver0_run_CUDA_kernel<nVarsFixed><<<blocksPerGrid, threadsPerBlock>>>(
                    DNDS_CUDA_DEVICE_VIEW_TMP_COPY(fv),
                    DNDS_CUDA_DEVICE_VIEW_TMP_COPY(u),
                    DNDS_CUDA_DEVICE_VIEW_TMP_COPY(u_grad));
        }
        else if (method == "pervar")
        {
            int v_size = nVarsFixed;
            int group_size_blk = std::min(32, v_size);

            int group_size = (v_size + group_size_blk - 1) / group_size_blk * group_size_blk;

            index total_threads = group_size * fv.mesh.NumCell();
            index cells_per_blk = (threadsPerBlock0 + group_size - 1) / group_size;

            index threadsPerBlock = group_size * cells_per_blk;
            index blocksPerGrid = (total_threads + threadsPerBlock - 1) / threadsPerBlock;

            // std::cout << "v_size " << v_size << std::endl;
            // std::cout << "group_size " << group_size << std::endl;
            // std::cout << "threadsPerBlock " << threadsPerBlock << std::endl;
            // std::cout << "blocksPerGrid " << blocksPerGrid << std::endl;
            for (int i = 0; i < nIter; i++)
                finiteVolumeCellOpTest_SOA_ver0_run_CUDA_kernel_var_per_thread<nVarsFixed><<<blocksPerGrid, threadsPerBlock>>>(
                    DNDS_CUDA_DEVICE_VIEW_TMP_COPY(fv),
                    DNDS_CUDA_DEVICE_VIEW_TMP_COPY(u),
                    DNDS_CUDA_DEVICE_VIEW_TMP_COPY(u_grad),
                    cells_per_blk, group_size);
        }
        else
            DNDS_assert_info(false, "no such method: " + method);

        cudaDeviceSynchronize();
    }

    template void finiteVolumeCellOpTest_SOA_ver0_entry<B, 1>::run(
        FiniteVolume::t_deviceView<B> &fv,
        std::array<tUDof<1>::t_deviceView<B>, 1> &u,
        std::array<tUGrad<1, 3>::t_deviceView<B>, 1> &u_grad,
        int nIter,
        const t_jsonconfig &settings);

    template void finiteVolumeCellOpTest_SOA_ver0_entry<B, 5>::run(
        FiniteVolume::t_deviceView<B> &fv,
        std::array<tUDof<1>::t_deviceView<B>, 5> &u,
        std::array<tUGrad<1, 3>::t_deviceView<B>, 5> &u_grad,
        int nIter,
        const t_jsonconfig &settings);
}