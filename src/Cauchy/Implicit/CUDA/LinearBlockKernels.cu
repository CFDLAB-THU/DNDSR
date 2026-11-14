#include "Cauchy/Implicit/CUDA/LinearBlockKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        /// One thread owns a complete row. Each block product and sparse sum
        /// follows column order; no floating-point atomics or parallel writes.
        template <int N, bool Jacobi, bool Zero, class InputRows>
        __global__ void LinearRows(LinearBlockInput request, InputRows rhs,
                                   InputRows owned, InputRows ghost, LinearWriteRows output)
        {
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x;
                 iRow < request.rows.RowCount(); iRow += index(gridDim.x) * blockDim.x)
            {
                const index iCell = request.rows(iRow, 0);
                real value[N]{};
                if constexpr (Jacobi)
                    for (int i = 0; i < N; ++i)
                        value[i] = rhs(iCell, i);
                else
                    for (int i = 0; i < N; ++i)
                    {
                        if (request.fullDiagonal)
                            for (int j = 0; j < N; ++j)
                                value[i] += request.diagonal(iCell, i + N * j) * owned(iCell, j);
                        else
                            value[i] = request.diagonal(iCell, i) * owned(iCell, i);
                    }
                if constexpr (!Zero)
                    for (index iBlock = 0; iBlock < request.columns.RowSize(iCell); ++iBlock)
                    {
                        const index column = request.columns(iCell, iBlock);
                        const auto block = request.blocks(iCell, iBlock);
                        real neighbor[N];
                        for (int j = 0; j < N; ++j)
                            neighbor[j] = column < request.ownedRows ? owned(column, j) : ghost(column - request.ownedRows, j);
                        for (int i = 0; i < N; ++i)
                        {
                            real product = 0;
                            for (int j = 0; j < N; ++j)
                                product += block[i + N * j] * neighbor[j];
                            if constexpr (Jacobi)
                                value[i] -= product;
                            else
                                value[i] += product;
                        }
                    }
                for (int i = 0; i < N; ++i)
                {
                    real result = value[i];
                    if constexpr (Jacobi)
                    {
                        result = 0;
                        if (request.fullDiagonal)
                            for (int j = 0; j < N; ++j)
                                result += request.diagonal(iCell, i + N * j) * value[j];
                        else
                            result = request.diagonal(iCell, i) * value[i];
                    }
                    output(iCell, i) = result;
                }
            }
        }
        template <class View>
        void CheckTable(View view, index rows, index width)
        {
            DNDS_check_throw_info(view.kind == R::RowLayoutKind::Table && view.RowCount() == rows && view.Width() == width,
                                  "CUDA linear operator requires matching combined AoS tables");
        }
        void CheckVector(VectorReadRows view, index rows, index width)
        {
            DNDS_check_throw_info(view.RowCount() == rows && view.Width() == width &&
                                      (view.kind == R::RowLayoutKind::Table ||
                                       (view.kind == R::RowLayoutKind::TiledColMajorTable && view.stride == 32)),
                                  "CUDA linear vector requires matching AoS or tiled32 input");
        }
        void Check(R::ExecutionContext context, const LinearBlockInput &request,
                   VectorReadRows rhs, VectorReadRows owned, VectorReadRows ghost,
                   LinearWriteRows output, bool jacobi, bool zero)
        {
            DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA &&
                                      request.components >= 5 && request.components <= 7 && request.ownedRows >= 0,
                                  "CUDA linear operator requires a CUDA context and 5/6/7 components");
            R::ValidateViewDevices(context.device, request.rows, request.diagonal, output);
            CheckTable(output, request.ownedRows, request.components);
            CheckTable(request.diagonal, request.ownedRows,
                       request.fullDiagonal ? request.components * request.components : request.components);
            DNDS_check_throw_info(!output.data || output.data != request.diagonal.data, "CUDA linear output aliases diagonal");
            if (jacobi)
            {
                rhs.ValidateDevice(context.device);
                CheckVector(rhs, request.ownedRows, request.components);
                DNDS_check_throw_info(!output.data || output.data != rhs.data, "CUDA Jacobi output aliases rhs");
            }
            if (!zero)
            {
                R::ValidateViewDevices(context.device, owned, request.columns);
                request.blocks.ValidateDevice(context.device);
                CheckVector(owned, request.ownedRows, request.components);
                DNDS_check_throw_info(request.columns.RowCount() == request.ownedRows &&
                                          request.blocks.RowNumber() == request.ownedRows && request.blocks.blockSize == request.components &&
                                          request.blocks.values.Width() == request.components * request.components,
                                      "CUDA linear block pattern shape differs");
                if (ghost.RowCount())
                {
                    ghost.ValidateDevice(context.device);
                    CheckVector(ghost, ghost.RowCount(), request.components);
                }
                DNDS_check_throw_info(!output.data || (output.data != owned.data && output.data != ghost.data &&
                                                       output.data != request.blocks.values.data),
                                      "CUDA linear output aliases an input");
            }
        }
        template <bool Jacobi, bool Zero>
        void Launch(R::ExecutionContext context, const LinearBlockInput &request, VectorReadRows rhs,
                    VectorReadRows owned, VectorReadRows ghost, LinearWriteRows output)
        {
            Check(context, request, rhs, owned, ghost, output, Jacobi, Zero);
            R::CheckKernelArgumentBudget<LinearBlockInput, VectorReadRows, VectorReadRows, VectorReadRows, LinearWriteRows>();
            if (!request.rows.RowCount())
                return;
            const unsigned blocks = unsigned(std::min<index>(65535, (request.rows.RowCount() + 127) / 128));
            const auto stream = static_cast<cudaStream_t>(context.stream);
            const auto issue = [&](auto right, auto own, auto remote)
            {
                if (request.components == 5)
                    LinearRows<5, Jacobi, Zero><<<blocks, 128, 0, stream>>>(request, right, own, remote, output);
                else if (request.components == 6)
                    LinearRows<6, Jacobi, Zero><<<blocks, 128, 0, stream>>>(request, right, own, remote, output);
                else
                    LinearRows<7, Jacobi, Zero><<<blocks, 128, 0, stream>>>(request, right, own, remote, output);
            };
            if (rhs.kind == R::RowLayoutKind::Table && owned.kind == R::RowLayoutKind::Table && ghost.kind == R::RowLayoutKind::Table)
            {
                const auto table = [](VectorReadRows view) -> LinearReadRows
                { return {view.data, view.offsets, view.rows, view.width, view.stride, view.sizes, view.kind, view.device}; };
                issue(table(rhs), table(owned), table(ghost));
            }
            else
                issue(rhs, owned, ghost);
            DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA linear block kernel launch failed");
        }
    }
    void ApplyLinearRows(R::ExecutionContext context, const LinearBlockInput &request,
                         VectorReadRows owned, VectorReadRows ghost, LinearWriteRows output)
    {
        Launch<false, false>(context, request, {}, owned, ghost, output);
    }
    void JacobiLinearRows(R::ExecutionContext context, const LinearBlockInput &request,
                          VectorReadRows rhs, VectorReadRows owned, VectorReadRows ghost,
                          LinearWriteRows output, bool zeroStart)
    {
        if (zeroStart)
            Launch<true, true>(context, request, rhs, {}, {}, output);
        else
            Launch<true, false>(context, request, rhs, owned, ghost, output);
    }
}
