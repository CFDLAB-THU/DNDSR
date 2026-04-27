#pragma once
/// @file Macros.hpp
/// @brief Project-wide preprocessor flags, branch-hint macros, and version/build
/// metadata strings. Included first by @ref Defines.hpp.

#include <string>

// #define EIGEN_USE_BLAS
// #define EIGEN_USE_LAPACKE_STRICT

/// @brief Enable Eigen's C++14 variable templates extension.
#define EIGEN_HAS_CXX14_VARIABLE_TEMPLATES 1

/// @brief Disable Eigen's own OpenMP parallelisation. DNDSR controls
/// threading explicitly through OpenMP parallel regions in the solvers.
#define EIGEN_DONT_PARALLELIZE 1

/// @brief Runtime-inspectable summary of which optional Eigen backends are compiled in.
static const std::string DNDS_Macros_State = std::string("DNDS_Macros ")
#ifdef EIGEN_USE_BLAS
                                             + " EIGEN_USE_BLAS "
#endif
#ifdef EIGEN_USE_LAPACKE_STRICT
                                             + " EIGEN_USE_LAPACKE_STRICT "
#endif
    ;

#if defined(__GNUC__) || defined(__clang__)
// GCC and Clang support __builtin_expect
/// @brief Branch-prediction hint: mark `x` as the likely branch. Maps to
/// `__builtin_expect(x, 1)` on GCC/Clang; no-op elsewhere.
#    define DNDS_likely(x) (__builtin_expect((x), 1))
/// @brief Branch-prediction hint: mark `x` as the unlikely branch. Maps to
/// `__builtin_expect(x, 0)` on GCC/Clang; no-op elsewhere.
#    define DNDS_unlikely(x) (__builtin_expect((x), 0))
#elif defined(_MSC_VER)
// MSVC does not support __builtin_expect
#    define DNDS_likely(x) (x)
#    define DNDS_unlikely(x) (x)
#else
// For other compilers, default to no-op
#    define DNDS_likely(x) (x)
#    define DNDS_unlikely(x) (x)
#endif

// experimental macros

#include "Experimentals.hpp"

/// @brief Runtime-inspectable build configuration summary: NDEBUG / NINSERT / experimental flags.
static const std::string DNDS_Defines_state =
    std::string("DNDS_Defines ") + DNDS_Macros_State + DNDS_Experimentals_State
#ifdef NDEBUG
    + " NDEBUG "
#else
    + " (no NDEBUG) "
#endif
#ifdef NINSERT
    + " NINSERT "
#else
    + " (no NINSERT) "
#endif
    ;

#ifndef DNDS_CURRENT_COMMIT_HASH
/// @brief Fallback when the build system did not inject the git hash.
#    define DNDS_CURRENT_COMMIT_HASH UNKNOWN
#endif

#ifndef DNDS_VERSION_STRING
/// @brief Fallback when the build system did not inject the version string.
#    define DNDS_VERSION_STRING "unknown"
#endif

/// @brief Stringize a macro value after a round of expansion. `DNDS_MACRO_TO_STRING(FOO)` yields the textual value of `FOO`.
#define DNDS_MACRO_TO_STRING(V) DNDS_str(V)
#define DNDS_str(V) #V

#if defined(__DNDS_REALLY_COMPILING__)
/// @brief Pick between "real compilation" and "IntelliSense preview" tokens.
/// @details IDEs that parse the project with `__DNDS_REALLY_COMPILING__` undefined
/// substitute the `intellisense` branch, letting us hide heavy template paths
/// from the editor's code model while keeping the `real` branch in actual builds.
#    define DNDS_SWITCH_INTELLISENSE(real, intellisense) real
#else
#    define DNDS_SWITCH_INTELLISENSE(real, intellisense) intellisense
#endif

// warning pragmas

#include "Warnings.hpp"
