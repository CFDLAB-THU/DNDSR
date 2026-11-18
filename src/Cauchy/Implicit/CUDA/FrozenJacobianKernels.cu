#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "DNDS/Algebra/BlockIndex.hpp"
#include "Cauchy/Implicit/CUDA/FrozenJacobianKernels.hpp"
#include "Cauchy/Implicit/CUDA/FrozenJacobianKernels_Point.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        template <int N>
        __device__ NumericalStatus AddPoint(const FrozenJacobianInput &s, index iCell,
                                            const JacobianPointRecord &point,
                                            const std::array<real, N> &input, std::array<real, N> &result)
        {
            std::array<real, N> state{}, derivative{};
            FlowState left{};
            Vector3 magnitudes{};
            for (int j = 0; j < N; ++j)
            {
                state[j] = s.means(point.other, j);
                derivative[j] = s.pressureDerivative(point.other, j);
            }
            if (s.parameters.roe)
            {
                for (int j = 0; j < 5; ++j)
                    left[j] = s.means(iCell, j);
                for (int j = 0; j < 3; ++j)
                    magnitudes[j] = s.magnitudes(point.sample, j);
            }
            return Detail::AddFrozenJacobianPoint<N>(s.parameters, iCell, point, left, state, derivative,
                                                     s.pressure(point.other, 0), s.radius(point.auxiliary, 0), s.radius(point.auxiliary, 1), magnitudes, input, result);
        }

        template <int N>
        __global__ void Assemble(FrozenJacobianInput s, LinearIndexRows selected,
                                 Algebra::BlockMatrixArrayView<real> output, FrozenJacobianStatus status)
        {
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x;
                 iRow < selected.RowCount(); iRow += index(gridDim.x) * blockDim.x)
            {
                const index iCell = selected(iRow, 0);
                NumericalStatus checked{};
                for (index iBlock = 0; iBlock < s.geometry.columns.RowSize(iCell) && checked; ++iBlock)
                {

                    for (int j = 0; j < N; ++j)
                    {
                        std::array<real, N> direction{}, product{};
                        direction[j] = 1;
                        for (const auto incidence : s.geometry.Points(iCell, iBlock))
                        {
                            checked = AddPoint<N>(s, iCell, s.geometry.Point(iCell, incidence), direction, product);
                            if (!checked)
                                break;
                        }
                        for (int i = 0; i < N; ++i)
                            output.Element(iCell, iBlock, i, j) = product[i];
                        if (!checked)
                            break;
                    }
                }
                status(iCell, 0) = checked;
            }
        }

        template <int N, bool Jacobi>
        __global__ void Apply(FrozenJacobianInput s, LinearIndexRows selected, MatrixReadRows diagonal,
                              bool fullDiagonal, VectorReadRows rhs, VectorReadRows owned,
                              VectorReadRows ghost, LinearWriteRows output, FrozenJacobianStatus status, bool rowMajorBlocks)
        {
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x;
                 iRow < selected.RowCount(); iRow += index(gridDim.x) * blockDim.x)
            {
                const index iCell = selected(iRow, 0);
                std::array<real, N> value{};
                NumericalStatus checked = status(iCell, 0);
                if (!checked)
                {
                    for (int i = 0; i < N; ++i)
                        output(iCell, i) = std::numeric_limits<real>::quiet_NaN();
                    continue;
                }
                for (int i = 0; i < N; ++i)
                    if constexpr (Jacobi)
                        value[i] = rhs(iCell, i);
                    else if (fullDiagonal)
                        for (int j = 0; j < N; ++j)
                            value[i] += diagonal(iCell, Algebra::DenseBlockEntry<N>(i, j, N, rowMajorBlocks)) * owned(iCell, j);
                    else
                        value[i] = diagonal(iCell, i) * owned(iCell, i);
                for (index iBlock = 0; iBlock < s.geometry.columns.RowSize(iCell) && checked; ++iBlock)
                {
                    const index column = s.geometry.columns(iCell, iBlock);
                    std::array<real, N> neighbor{}, product{};
                    for (int j = 0; j < N; ++j)
                        neighbor[j] = column < owned.RowCount() ? owned(column, j) : ghost(column - owned.RowCount(), j);
                    for (const auto incidence : s.geometry.Points(iCell, iBlock))
                    {
                        checked = AddPoint<N>(s, iCell, s.geometry.Point(iCell, incidence), neighbor, product);
                        if (!checked)
                            break;
                    }
                    for (int i = 0; i < N; ++i)
                        if constexpr (Jacobi)
                            value[i] -= product[i];
                        else
                            value[i] += product[i];
                }
                if (checked)
                    for (int i = 0; i < N; ++i)
                    {
                        real result = value[i];
                        if constexpr (Jacobi)
                        {
                            result = 0;
                            if (fullDiagonal)
                                for (int j = 0; j < N; ++j)
                                    result += diagonal(iCell, Algebra::DenseBlockEntry<N>(i, j, N, rowMajorBlocks)) * value[j];
                            else
                                result = diagonal(iCell, i) * value[i];
                        }
                        output(iCell, i) = result;
                        if (!std::isfinite(result))
                            checked = NumericalStatus{NumericalError::InvalidState, 3, iCell};
                    }
                // A deferred numerical observation must never mistake an old
                // finite output row for a successful current action. Status is
                // authoritative; poisoning also makes later norm/finite checks
                // reject an accidentally consumed failed result.
                if (!checked)
                    for (int i = 0; i < N; ++i)
                        output(iCell, i) = std::numeric_limits<real>::quiet_NaN();
                status(iCell, 0) = checked;
            }
        }
        template <class View>
        void Table(View view, index rows, index width)
        {
            DNDS_check_throw_info((view.kind == R::RowLayoutKind::Table || view.kind == R::RowLayoutKind::ColMajorTable || view.kind == R::RowLayoutKind::TiledColMajorTable) && view.RowCount() == rows && view.Width() == width,
                                  "frozen CUDA Jacobian requires matching AoS tables");
        }
        void Vector(VectorReadRows view, index rows, index width)
        {
            DNDS_check_throw_info(view.RowCount() == rows && view.Width() == width &&
                                      (view.kind == R::RowLayoutKind::Table ||
                                       (view.kind == R::RowLayoutKind::TiledColMajorTable && view.stride == 32)),
                                  "frozen CUDA vector requires matching AoS or tiled32 input");
        }
        void Check(R::ExecutionContext context, const FrozenJacobianInput &s, LinearIndexRows selected, FrozenJacobianStatus status)
        {
            const auto &p = s.parameters;
            DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && p.components >= 5 && p.components <= 7 &&
                                      std::isfinite(p.scale) && p.scale >= 0 && (!p.roe || p.components == 5),
                                  "unsupported frozen CUDA Jacobian profile");
            s.geometry.ValidateDevice(context.device);
            R::ValidateViewDevices(context.device, s.means, s.pressure, s.pressureDerivative, s.radius, selected, status);
            Vector(s.means, s.means.RowCount(), p.components);
            Table(s.pressure, s.means.RowCount(), 1);
            Table(s.pressureDerivative, s.means.RowCount(), p.components);
            Table(s.radius, s.radius.RowCount(), 2);
            DNDS_check_throw_info(status.RowCount() == s.geometry.columns.RowCount() && s.means.RowCount() >= status.RowCount() &&
                                      s.geometry.blockOffsets.RowCount() == status.RowCount() + 1,
                                  "frozen CUDA Jacobian row spaces differ");
            if (p.roe)
            {
                s.magnitudes.ValidateDevice(context.device);
                Table(s.magnitudes, s.magnitudes.RowCount(), 3);
                p.gas.Validate();
            }
        }
    }
    void AssembleFrozenJacobian(R::ExecutionContext context, const FrozenJacobianInput &s, LinearIndexRows selected,
                                Algebra::BlockMatrixArrayView<real> output, FrozenJacobianStatus status)
    {
        Check(context, s, selected, status);
        output.ValidateDevice(context.device);
        DNDS_check_throw_info(output.RowNumber() == s.geometry.columns.RowCount() && output.blockSize == s.parameters.components,
                              "frozen CUDA Jacobian block shape differs");
        Table(output.values, s.geometry.incidences.RowCount(), s.parameters.components * s.parameters.components);
        DNDS_check_throw_info(!output.values.data || (output.values.data != s.means.data && output.values.data != s.pressure.data &&
                                                      output.values.data != s.pressureDerivative.data && output.values.data != s.radius.data &&
                                                      output.values.data != s.magnitudes.data),
                              "frozen CUDA Jacobian blocks alias a snapshot input");
        R::CheckKernelArgumentBudget<FrozenJacobianInput, LinearIndexRows, Algebra::BlockMatrixArrayView<real>, FrozenJacobianStatus>();
        if (!selected.RowCount())
            return;
        const unsigned blocks = unsigned(std::min<index>(65535, (selected.RowCount() + 127) / 128));
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        if (s.parameters.components == 5)
            Assemble<5><<<blocks, 128, 0, stream>>>(s, selected, output, status);
        else if (s.parameters.components == 6)
            Assemble<6><<<blocks, 128, 0, stream>>>(s, selected, output, status);
        else
            Assemble<7><<<blocks, 128, 0, stream>>>(s, selected, output, status);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "frozen CUDA Jacobian assembly launch failed");
    }
    void ApplyFrozenJacobian(R::ExecutionContext context, const FrozenJacobianInput &s, LinearIndexRows selected,
                             MatrixReadRows diagonal, bool fullDiagonal, VectorReadRows rhs, VectorReadRows owned,
                             VectorReadRows ghost, LinearWriteRows output, FrozenJacobianStatus status, bool jacobi, bool rowMajorBlocks)
    {
        Check(context, s, selected, status);
        R::ValidateViewDevices(context.device, diagonal, owned, output);
        const index rows = s.geometry.columns.RowCount(), n = s.parameters.components;
        Table(diagonal, rows, fullDiagonal ? n * n : n);
        Vector(owned, rows, n);
        Table(output, rows, n);
        if (ghost.RowCount())
        {
            ghost.ValidateDevice(context.device);
            Vector(ghost, s.means.RowCount() - rows, n);
        }
        if (jacobi)
        {
            rhs.ValidateDevice(context.device);
            Vector(rhs, rows, n);
        }
        DNDS_check_throw_info(!output.data || (output.data != diagonal.data && output.data != owned.data &&
                                               output.data != ghost.data && (!jacobi || output.data != rhs.data) && output.data != s.means.data &&
                                               output.data != s.pressure.data && output.data != s.pressureDerivative.data &&
                                               output.data != s.radius.data && output.data != s.magnitudes.data),
                              "frozen CUDA Jacobian output aliases an input");
        R::CheckKernelArgumentBudget<FrozenJacobianInput, LinearIndexRows, MatrixReadRows, bool, VectorReadRows,
                                     VectorReadRows, VectorReadRows, LinearWriteRows, FrozenJacobianStatus, bool>();
        if (!selected.RowCount())
            return;
        const unsigned blocks = unsigned(std::min<index>(65535, (selected.RowCount() + 127) / 128));
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        if (jacobi)
        {
            if (n == 5)
                Apply<5, true><<<blocks, 128, 0, stream>>>(s, selected, diagonal, fullDiagonal, rhs, owned, ghost, output, status, rowMajorBlocks);
            else if (n == 6)
                Apply<6, true><<<blocks, 128, 0, stream>>>(s, selected, diagonal, fullDiagonal, rhs, owned, ghost, output, status, rowMajorBlocks);
            else
                Apply<7, true><<<blocks, 128, 0, stream>>>(s, selected, diagonal, fullDiagonal, rhs, owned, ghost, output, status, rowMajorBlocks);
        }
        else
        {
            if (n == 5)
                Apply<5, false><<<blocks, 128, 0, stream>>>(s, selected, diagonal, fullDiagonal, {}, owned, ghost, output, status, rowMajorBlocks);
            else if (n == 6)
                Apply<6, false><<<blocks, 128, 0, stream>>>(s, selected, diagonal, fullDiagonal, {}, owned, ghost, output, status, rowMajorBlocks);
            else
                Apply<7, false><<<blocks, 128, 0, stream>>>(s, selected, diagonal, fullDiagonal, {}, owned, ghost, output, status, rowMajorBlocks);
        }
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "frozen CUDA Jacobian action launch failed");
    }
}
