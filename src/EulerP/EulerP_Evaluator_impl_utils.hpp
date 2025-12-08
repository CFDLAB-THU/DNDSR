#pragma once

#include "DNDS/DeviceStorage.hpp"
#include "DNDS/CUDA_Utils.hpp"
namespace DNDS::EulerP::detail
{
    struct FLocalAccessor_noOp
    {
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(FLocalAccessor_noOp, FLocalAccessor_noOp)
        real dummy_;
        DNDS_FORCEINLINE DNDS_DEVICE real &operator()(int i) { return dummy_; }
    };

    struct FGlobalAccessor_noOp
    {
        real dummy_;
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(FGlobalAccessor_noOp, FGlobalAccessor_noOp)
        DNDS_FORCEINLINE DNDS_DEVICE real &operator()(index iPnt, int i) { return dummy_; }
    };

    template <int local_stride_fixed, int max_tid_fixed,
              class TFLocalAccessor, class TFGlobalAccessor,
              int bufferSize_idx, int bufferSize_val>
    DNDS_FORCEINLINE DNDS_DEVICE void CUDA_Local2GlobalAssign(
        TFLocalAccessor &&FLocalAccessor,
        TFGlobalAccessor &&FGLobalAccessor,
        CUDA::SharedBuffer<index, bufferSize_idx> &shared_buf_idx,
        CUDA::SharedBuffer<real, bufferSize_val> &shared_buf_val,
        index iPnt, index iPntMax)
    {
#ifndef __CUDA_ARCH__
        static_assert(local_stride_fixed > 0 && local_stride_fixed < 0);
#endif
        static_assert(local_stride_fixed > 0);
        static_assert(max_tid_fixed > 0);
        static_assert(bufferSize_idx >= max_tid_fixed);
        // TODO: support dynamic sized?

        constexpr int local_stride = local_stride_fixed;
        constexpr int local_stride_buf = (local_stride / 2) * 2 + 1;
        static_assert(bufferSize_val >= local_stride_buf * max_tid_fixed);

        real *buf_data = shared_buf_val.buffer;
        index *iPntThread = shared_buf_idx.buffer;

        int tid = CUDA::tid_x();
        int bDim = CUDA::bDim_x();
        DNDS_HD_assert(tid < max_tid_fixed && tid >= 0);
        iPntThread[tid] = iPnt; //! can out of bounds
        for (int i = 0; i < local_stride; i++)
            buf_data[tid * local_stride_buf + i] = FLocalAccessor(i);

        CUDA::sync_threads();
        for (int i = 0; i < local_stride; i++)
        {
            int iComp = (i * bDim + tid);
            int iPntInBlock = iComp / local_stride;
            int iCompSub = iComp % local_stride;
            int iCompBuf = (local_stride == local_stride_buf) ? iComp : (iPntInBlock * local_stride_buf + iCompSub);
            // int iComp = (i * bDim + tid) % local_stride;
            index iPntC = iPntThread[iPntInBlock];
            if (iPntC < iPntMax)
                FGLobalAccessor(iPntC, iCompSub) = buf_data[iCompBuf];
        }
    }
}