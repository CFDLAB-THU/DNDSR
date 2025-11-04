#pragma once
#include <string>

// #define EIGEN_USE_BLAS
// #define EIGEN_USE_LAPACKE_STRICT

#define EIGEN_HAS_CXX14_VARIABLE_TEMPLATES 1

#define EIGEN_DONT_PARALLELIZE 1

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
#    define DNDS_likely(x) (__builtin_expect((x), 1))
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
#    define DNDS_CURRENT_COMMIT_HASH UNKNOWN
#endif

#define DNDS_MACRO_TO_STRING(V) __DNDS_str(V)
#define __DNDS_str(V) #V

#if defined(__DNDS_REALLY_COMPILING__)
#    define DNDS_SWITCH_INTELLISENSE(real, intellisense) real
#else
#    define DNDS_SWITCH_INTELLISENSE(real, intellisense) intellisense
#endif

// warning pragmas

#include "Warnings.hpp"
