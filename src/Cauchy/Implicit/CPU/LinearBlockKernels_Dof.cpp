#include "Cauchy/Implicit/CPU/LinearBlockKernels.hpp"
#include "Cauchy/Implicit/CPU/LinearBlockKernels_Impl.hpp"

namespace DNDS::Cauchy::CPU
{
    void ApplyLinearRows(const LinearBlockInput &request, LinearDofRead owned, LinearDofRead ghost, LinearDofWrite output)
    {
        WithLinearMatrixRows(request.diagonal, request.rowMajorBlocks, [&](auto diagonalRows)
                             {
        const auto diagonal = Algebra::MakeEigenRowMatrices(diagonalRows, request.components, request.fullDiagonal ? request.components : 1, Runtime::Device::Host(), request.rowMajorBlocks);
        request.blocks.WithActions([&](const auto &blocks)
                                   { ApplyBlockRows(owned, ghost, output, diagonal,
                                                    blocks, request.pattern, request.rows, request.components, request.fullDiagonal); }); });
    }
    void JacobiLinearRows(const LinearBlockInput &request, LinearDofRead rhs, LinearDofRead owned, LinearDofRead ghost, LinearDofWrite output, bool zeroStart)
    {
        WithLinearMatrixRows(request.diagonal, request.rowMajorBlocks, [&](auto diagonalRows)
                             {
        const auto inverse = Algebra::MakeEigenRowMatrices(diagonalRows, request.components, request.fullDiagonal ? request.components : 1, Runtime::Device::Host(), request.rowMajorBlocks);
        if (zeroStart)
        {
            JacobiRows<true>(rhs, rhs, rhs, output, inverse, 0,
                             request.pattern, request.rows, request.components, request.fullDiagonal);
            return;
        }
        request.blocks.WithActions([&](const auto &blocks)
                                   {
            const auto invoke = [&](auto zero)
            {
                JacobiRows<decltype(zero)::value>(rhs, owned, ghost, output,
                    inverse, blocks, request.pattern, request.rows, request.components, request.fullDiagonal);
            };
            if (zeroStart) invoke(std::true_type{}); else invoke(std::false_type{}); }); });
    }
    void LowerSGSRows(const LinearBlockInput &request, LinearDofRead rhs, LinearDofRead old, LinearDofRead ghost, LinearDofWrite output, bool zeroStart)
    {
        WithLinearMatrixRows(request.diagonal, request.rowMajorBlocks, [&](auto diagonalRows)
                             {
        const auto inverse = Algebra::MakeEigenRowMatrices(diagonalRows, request.components, request.fullDiagonal ? request.components : 1, Runtime::Device::Host(), request.rowMajorBlocks);
        request.blocks.WithActions([&](const auto &blocks)
                                   {
            const auto invoke = [&](auto zero)
            {
                constexpr bool Zero = decltype(zero)::value;
                if (request.components == 5)
                    ScanSGSRows<false, 5, Zero>(rhs, old, output, output, ghost, inverse,
                        blocks, request.pattern, request.rows, request.components, request.fullDiagonal, request.sweepPartitions);
                else
                    ScanSGSRows<false, Eigen::Dynamic, Zero>(rhs, old, output, output, ghost, inverse,
                        blocks, request.pattern, request.rows, request.components, request.fullDiagonal, request.sweepPartitions);
            };
            if (zeroStart) invoke(std::true_type{}); else invoke(std::false_type{}); }); });
    }
    void UpperSGSRows(const LinearBlockInput &request, LinearDofRead rhs, LinearDofRead old, LinearDofRead forward, LinearDofRead ghost, LinearDofWrite output, bool zeroStart)
    {
        WithLinearMatrixRows(request.diagonal, request.rowMajorBlocks, [&](auto diagonalRows)
                             {
        const auto inverse = Algebra::MakeEigenRowMatrices(diagonalRows, request.components, request.fullDiagonal ? request.components : 1, Runtime::Device::Host(), request.rowMajorBlocks);
        request.blocks.WithActions([&](const auto &blocks)
                                   {
            const auto invoke = [&](auto zero)
            {
                constexpr bool Zero = decltype(zero)::value;
                if (request.components == 5)
                    ScanSGSRows<true, 5, Zero>(rhs, old, forward, output, ghost, inverse,
                        blocks, request.pattern, request.rows, request.components, request.fullDiagonal, request.sweepPartitions);
                else
                    ScanSGSRows<true, Eigen::Dynamic, Zero>(rhs, old, forward, output, ghost, inverse,
                        blocks, request.pattern, request.rows, request.components, request.fullDiagonal, request.sweepPartitions);
            };
            if (zeroStart) invoke(std::true_type{}); else invoke(std::false_type{}); }); });
    }
}
