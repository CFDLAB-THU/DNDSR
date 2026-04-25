/**
 * @file EulerP_Evaluator_impl_utils.hpp
 * @brief Internal utility types for the EulerP Evaluator kernel implementations.
 *
 * Provides:
 * - @c FLocalAccessor_noOp / @c FGlobalAccessor_noOp: No-op accessor stubs used as template
 *   arguments when no local/global accumulation is needed in a kernel.
 * - @c CUDA_Local2GlobalAssign: CUDA shared-memory helper for bank-conflict-free coalesced
 *   write-back from thread-local storage to global memory.
 *
 * @note This header is an internal implementation detail of the EulerP Evaluator.
 *       Contents reside in the @c DNDS::EulerP::detail namespace.
 */
#pragma once

#include "DNDS/Device/DeviceStorage.hpp"
#include "DNDS/Device/CUDA_Utils.hpp"

/**
 * @brief Internal implementation detail namespace for EulerP Evaluator utilities.
 */
namespace DNDS::EulerP::detail
{
    /**
     * @brief No-op local accessor returning a dummy real value.
     *
     * Used as a template argument for kernels that do not require
     * thread-local accumulation. All writes go to a discarded dummy member.
     */
    struct FLocalAccessor_noOp
    {
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(FLocalAccessor_noOp, FLocalAccessor_noOp)
        real dummy_; ///< Discarded dummy value; all accesses read/write this.

        /// @brief Returns a reference to the dummy value (ignores index).
        /// @param i Component index (unused).
        /// @return Reference to @c dummy_.
        DNDS_FORCEINLINE DNDS_DEVICE real &operator()(int i) { return dummy_; }
    };

    /**
     * @brief No-op global accessor returning a dummy real value.
     *
     * Used as a template argument for kernels that do not require
     * global memory write-back. All writes go to a discarded dummy member.
     */
    struct FGlobalAccessor_noOp
    {
        real dummy_; ///< Discarded dummy value; all accesses read/write this.
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(FGlobalAccessor_noOp, FGlobalAccessor_noOp)

        /// @brief Returns a reference to the dummy value (ignores point and component indices).
        /// @param iPnt Point index (unused).
        /// @param i Component index (unused).
        /// @return Reference to @c dummy_.
        DNDS_FORCEINLINE DNDS_DEVICE real &operator()(index iPnt, int i) { return dummy_; }
    };

#ifdef DNDS_USE_CUDA

    /**
     * @brief CUDA shared-memory helper for bank-conflict-free coalesced write-back.
     *
     * Transfers per-thread local accumulation buffers to global memory through CUDA
     * shared memory, using a stride-padded layout to avoid shared memory bank conflicts.
     *
     * The algorithm:
     * 1. Each thread writes its @c local_stride_fixed elements into shared memory at
     *    a padded stride @c local_stride_buf = (local_stride/2)*2+1 to avoid bank conflicts.
     * 2. After a thread synchronization barrier, threads cooperatively read from shared
     *    memory and write to global memory in a coalesced pattern where thread @c tid
     *    writes element @c (i * blockDim + tid) to the appropriate global location.
     *
     * @tparam local_stride_fixed Number of elements per point (must be > 0, compile-time constant).
     * @tparam max_tid_fixed Maximum number of threads per block (must be > 0, compile-time constant).
     * @tparam TFLocalAccessor Thread-local accessor type: operator()(int i) -> real&.
     * @tparam TFGlobalAccessor Global accessor type: operator()(index iPnt, int i) -> real&.
     * @tparam bufferSize_idx Size of the shared index buffer (must be >= max_tid_fixed).
     * @tparam bufferSize_val Size of the shared value buffer (must be >= local_stride_buf * max_tid_fixed).
     * @param FLocalAccessor Functor providing access to thread-local data.
     * @param FGLobalAccessor Functor providing access to global output data.
     * @param shared_buf_idx Shared memory buffer for point indices.
     * @param shared_buf_val Shared memory buffer for intermediate values.
     * @param iPnt Global point index for this thread.
     * @param iPntMax Total number of valid points (writes beyond this are skipped).
     */
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
#    ifndef __CUDA_ARCH__
        static_assert(local_stride_fixed > 0 && local_stride_fixed < 0);
#    endif
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
#endif
}