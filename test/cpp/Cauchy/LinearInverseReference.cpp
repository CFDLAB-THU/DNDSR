#include "LinearInverseReference.hpp"
#include "Solver/BlockInverse.hpp"

namespace CauchyTest
{
    DNDS::real PartialPivotReciprocalCondition(const Eigen::MatrixXd &matrix)
    {
        return matrix.partialPivLu().rcond();
    }
    DNDS::NumericalResult<Eigen::MatrixXd> DiagonalInverseReference(const Eigen::MatrixXd &matrix)
    {
        auto result = DNDS::Direct::TryInvertDiagonalBlock(matrix);
        if (!result)
            return result.status;
        return std::move(result.value.value);
    }
}
