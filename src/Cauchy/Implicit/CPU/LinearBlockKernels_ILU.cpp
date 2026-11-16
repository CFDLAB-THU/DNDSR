#include "Cauchy/Implicit/CPU/LinearBlockKernels.hpp"
#include "Cauchy/Implicit/CPU/LinearBlockKernels_Impl.hpp"
namespace DNDS::Cauchy::CPU
{
    void FillLinearFactor(LinearDiagonalRows diagonal, OffDiagonalView blocks, Algebra::BlockMatrixView<real> factor,
                          LinearColumnsView sourceBlocks,
                          index components, bool fullDiagonal, bool rowMajorBlocks)
    {
        const auto d = Algebra::MakeEigenRowMatrices(diagonal, components, fullDiagonal ? components : 1, Runtime::Device::Host(), rowMajorBlocks);
        blocks.WithBlocks([&](const auto &values)
                          { FillILUBlocks(d, values, factor, sourceBlocks, fullDiagonal); });
    }
}
