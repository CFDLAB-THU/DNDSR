#pragma once

#include "DNDS/Core/NumericalStatus.hpp"
#include <Eigen/Core>

namespace CauchyTest
{
    /// Host-only Eigen oracle, deliberately compiled by the C++ compiler.
    /// CUDA test/benchmark TUs never instantiate Eigen's host-only LU internals.
    DNDS::real PartialPivotReciprocalCondition(const Eigen::MatrixXd &matrix);
    DNDS::NumericalResult<Eigen::MatrixXd> DiagonalInverseReference(const Eigen::MatrixXd &matrix);
}
