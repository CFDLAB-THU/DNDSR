#pragma once
/// @file EigenPCH.hpp
/// @brief Pre-compiled-header style shim that includes the heavy Eigen headers
/// under DNDSR's warning suppressions. Every DNDS file that touches Eigen pulls
/// this in instead of `<Eigen/Core>` directly.

#include "Warnings.hpp"
DISABLE_WARNING_PUSH
DISABLE_WARNING_MAYBE_UNINITIALIZED
DISABLE_WARNING_CLASS_MEMACCESS
#include <Eigen/Core>
#include <Eigen/Dense> //?It seems Mat.determinant() would be undefined rather than undeclared...
#include <Eigen/Sparse>
DISABLE_WARNING_POP