#pragma once
#include <string>

// #define EIGEN_USE_BLAS
// #define EIGEN_USE_LAPACKE_STRICT

#define EIGEN_HAS_CXX14_VARIABLE_TEMPLATES 1

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
#define DNDS_likely(x) (__builtin_expect((x), 1))
#define DNDS_unlikely(x) (__builtin_expect((x), 0))
#elif defined(_MSC_VER)
// MSVC does not support __builtin_expect
#define DNDS_likely(x) (x)
#define DNDS_unlikely(x) (x)
#else
// For other compilers, default to no-op
#define DNDS_likely(x) (x)
#define DNDS_unlikely(x) (x)
#endif
