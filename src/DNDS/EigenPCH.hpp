#pragma once

#include "Warnings.hpp"
DISABLE_WARNING_PUSH
DISABLE_WARNING_MAYBE_UNINITIALIZED
#include <Eigen/Core>
#include <Eigen/Dense> //?It seems Mat.determinant() would be undefined rather than undeclared...
#include <Eigen/Sparse>
DISABLE_WARNING_POP