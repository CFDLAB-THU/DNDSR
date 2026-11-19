#include "Cauchy/Implicit/CUDA/AMGGeometry.hpp"
#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include <cuda_runtime.h>
#include <limits>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        template <class T>
        using Column = R::ArrayView<T, R::AccessBoundsChecks, R::FixedRows<1>>;

        /// Each thread exclusively owns one CSR row, including the additions
        /// from duplicate physical connections. Evaluate each contribution in
        /// parent zero's frame on both traversals, also for rotated periodicity.
        __global__ void FaceStrengths(Column<const FiniteVolumeStencilView> geometry,
                                      Column<const int> offsets, Column<const int> columns,
                                      Column<real> weights, Column<int> error)
        {
            const auto &s = geometry(0, 0);
            const index rows = offsets.RowCount() - 1;
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < rows; iCell += index(gridDim.x) * blockDim.x)
            {
                const index begin = offsets(iCell, 0), end = offsets(iCell + 1, 0);
                for (index slot = begin; slot < end; ++slot)
                    weights(slot, 0) = 0;
                const auto faces = s.topology.CellFaces(iCell);
                for (rowsize i = 0; i < faces.size(); ++i)
                {
                    const index face = faces[i];
                    const auto cells = s.topology.FaceCells(face);
                    const int side = s.topology.CellIsFaceBack(iCell, face, i) ? 0 : 1;
                    const index other = cells[1 - side];
                    if (other == UnInitIndex || other == iCell)
                        continue;
                    const index slot = s.columns(face, side);
                    if (cells[side] != iCell || other < 0 || other >= rows || slot < 0 ||
                        slot >= end - begin || columns(begin + slot, 0) != other)
                    {
                        atomicOr(error.data, 1);
                        continue;
                    }
                    const real distance = (s.metric.GetNeighborCent(face, 0) - s.metric.GetCellCent(cells[0])).norm();
                    const real area = s.metric.GetFaceArea(face);
                    if (!(distance > 0 && isfinite(distance) && area > 0 && isfinite(area)))
                    {
                        atomicOr(error.data, 2);
                        continue;
                    }
                    weights(begin + slot, 0) += area / distance;
                }
            }
        }

        /// The endpoints may enumerate multiple connecting faces in different
        /// orders. Allow rounding in their sums while rejecting absent, empty
        /// or asymmetric edges before the matching selector sees the graph.
        __global__ void CheckReciprocals(Column<const int> offsets, Column<const int> columns,
                                         Column<const real> weights, Column<int> error)
        {
            const index rows = offsets.RowCount() - 1;
            for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < rows; iCell += index(gridDim.x) * blockDim.x)
                for (index k = offsets(iCell, 0); k < offsets(iCell + 1, 0); ++k)
                {
                    const index other = columns(k, 0);
                    const real w = weights(k, 0);
                    bool reciprocal = false;
                    if (other >= 0 && other < rows && other != iCell && isfinite(w) && w > 0)
                        for (index j = offsets(other, 0); j < offsets(other + 1, 0); ++j)
                            if (columns(j, 0) == iCell)
                            {
                                const real back = weights(j, 0);
                                reciprocal = isfinite(back) && fabs(w - back) <= 64 * std::numeric_limits<real>::epsilon() * fmax(w, back);
                                break;
                            }
                    if (!reciprocal)
                        atomicOr(error.data, 4);
                }
        }
    }

    R::Array<real> BuildAMGGeometricWeights(R::ExecutionContext context, const FiniteVolumeStencilStorage::ReadAccess &stencil,
                                            const LinearPatternStorage &pattern)
    {
        const auto descriptor = stencil.KernelDescriptor();
        const auto &offsets = pattern.Offsets32(), &columns = pattern.Columns32();
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && descriptor.GetDevice() == context.device &&
                                  offsets.GetDevice() == context.device && columns.GetDevice() == context.device &&
                                  offsets.RowCount() == stencil.View().jacobianOffsets.RowCount(),
                              "AmgX geometry and matrix row spaces differ");
        R::Array<real> weights(R::RowLayout::Dense(columns.RowCount(), 1), context.device);
        weights.SetName("Cauchy/linear/amgx/geometricWeights", DNDS_ALLOCATION_SITE());
        const index rows = offsets.RowCount() - 1;
        if (!rows)
            return weights;
        const auto stream = R::CUDAStream(context);
        const unsigned blocks = unsigned(std::min<index>(1024, (rows + 127) / 128));
        R::Array<int> error(R::RowLayout::Dense(1, 1), context.device);
        {
            const auto offRead = offsets.Read(context), colRead = columns.Read(context);
            const auto off = offRead.View<R::FixedRows<1>>(), col = colRead.View<R::FixedRows<1>>();
            auto weightWrite = weights.Write(context);
            auto errorWrite = error.Write(context);
            const auto result = weightWrite.View<R::FixedRows<1>>();
            const auto flag = errorWrite.View<R::FixedRows<1>>();
            DNDS_check_throw_info(cudaMemsetAsync(flag.data, 0, sizeof(int), stream) == cudaSuccess, "AmgX geometry status initialization failed");
            FaceStrengths<<<blocks, 128, 0, stream>>>(descriptor, off, col, result, flag);
            DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "AmgX face conductance kernel launch failed");
            CheckReciprocals<<<blocks, 128, 0, stream>>>(off, col, result.ConstView(), flag);
            DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "AmgX reciprocal conductance check launch failed");
        }
        const auto hostError = error.To(R::Device::Host(), context);
        const int invalid = hostError.Read().View<R::FixedRows<1>>()(0, 0);
        DNDS_check_throw_info(!(invalid & 1), "AmgX geometry face slot does not match the local matrix column");
        DNDS_check_throw_info(!(invalid & 2), "AmgX interior geometry has a nonpositive or nonfinite distance or area");
        DNDS_check_throw_info(!(invalid & 4), "AmgX geometric CSR strengths must be positive, finite and reciprocal");
        return weights;
    }
}
