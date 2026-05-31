#pragma once
/// @file Errors.hpp
/// @brief Assertion / error-handling macros and supporting helper functions.
///
/// ## Overview
/// Three distinct families of checks are provided; choose based on how the
/// failure should surface:
///
/// | Macro                     | Release behaviour                 | Failure mode              |
/// |---------------------------|-----------------------------------|---------------------------|
/// | @ref DNDS_assert               | Always active (MAX level, unless NDEBUG) | `std::abort()`       |
/// | @ref DNDS_assert_info          | Always active (MAX level, unless NDEBUG) | `std::abort()` + msg |
/// | @ref DNDS_assert_infof         | Always active (MAX level, unless NDEBUG) | `std::abort()` + fmt |
/// | @ref DNDS_assert_l             | Level-dependent (see below)       | `std::abort()`            |
/// | @ref DNDS_assert_info_l        | Level-dependent (see below)       | `std::abort()` + message  |
/// | @ref DNDS_assert_infof_l       | Level-dependent (see below)       | `std::abort()` + fmtprintf|
/// | @ref DNDS_check_throw          | Always active                     | `throw std::runtime_error`|
/// | @ref DNDS_check_throw_info     | Always active                     | `throw` + message         |
/// | @ref DNDS_HD_assert            | MAX level; disabled by NDEBUG / NDEBUG_DEVICE | abort / trap |
/// | @ref DNDS_HD_assert_infof      | MAX level; disabled by NDEBUG / NDEBUG_DEVICE | abort / trap + fmt |
/// | @ref DNDS_HD_assert_l          | Level-dependent (see below)       | abort / trap              |
/// | @ref DNDS_HD_assert_infof_l    | Level-dependent (see below)       | abort / trap + fmt        |
///
/// Prefer @ref DNDS_assert for hard invariants that must never fail; use
/// the leveled `_l` variants (levels 0..DNDS_ASSERT_LEVEL_MAX-1) for checks
/// that can be stripped in release builds via `-DDNDS_ASSERT_LEVEL=N`.
/// Use @ref DNDS_check_throw for user-input / runtime validation where a
/// recoverable exception is preferred over abort.
///
/// The device variants (`DNDS_HD_*`) expand to host asserts on the host and to
/// atomic-guarded PTX `trap` on CUDA devices so only one thread prints.

#include "Macros.hpp"

// assert macros

#include <iostream>
#include <cstdarg>
#include <array>
#include <sstream>
namespace DNDS
{
    /// @brief Return a symbolicated stack trace for the calling thread.
    /// @details Host-only, implemented with `boost::stacktrace` (or similar).
    /// Used by the `assert_false*` helpers below.
    std::string getTraceString();

    /// @brief Low-level: print a red "DNDS_assertion failed" line and abort.
    inline void assert_false(const char *expr, const char *file, int line)
    {
        std::cerr << getTraceString() << "\n";
        std::cerr << "\033[91m DNDS_assertion failed\033[39m: \"" << expr << "\"  at [  " << file << ":" << line << "  ]" << std::endl;
        std::abort();
    }

    /// @brief Variant of #assert_false that prints an extra `info` string.
    inline void assert_false_info(const char *expr, const char *file, int line, const std::string &info)
    {
        std::cerr << getTraceString() << "\n";
        std::cerr << "\033[91m DNDS_assertion failed\033[39m: \"" << expr << "\"  at [  " << file << ":" << line << "  ]\n"
                  << info << std::endl;
        std::abort();
    }

    /// @brief `printf`-style variant of #assert_false. Used by @ref DNDS_assert_infof.
    inline void assert_false_infof(const char *expr, const char *file, int line,
                                   const char *info, ...)
    {
        va_list args;
        va_start(args, info);
        std::cerr << getTraceString() << "\n";
        std::cerr << "\033[91m DNDS_assertion failed\033[39m: \"" << expr << "\"  at [  " << file << ":" << line << "  ]\n";
        // Compile-time constant 1024 * 512 = 524288 fits in int32_t;
        // no runtime overflow is possible.
        // NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result)
        std::array<char, 1024 * 512> format_buf{};
        std::vsnprintf(format_buf.data(), format_buf.size(), info, args);
        va_end(args);
        std::cerr << format_buf.data() << std::endl;
        std::abort();
    }

    /// @brief Throwing variant of #assert_false_info. Used by @ref DNDS_check_throw.
    /// @tparam TException Exception type to throw (defaults to `std::runtime_error`).
    /// Currently the implementation ignores the template parameter and always
    /// throws `std::runtime_error`; kept for future customisation.
    template <class TException = std::runtime_error>
    void assert_false_info_throw(const char *expr, const char *file, int line, const std::string &info)
    {
        std::stringstream ss;
        ss << getTraceString() << "\n";
        ss << "\033[91m DNDS_assertion failed\033[39m: \"" << expr << "\"  at [  " << file << ":" << line << "  ]\n"
           << info << std::endl;
        throw std::runtime_error(ss.str());
    }
}

/// @brief Runtime check active in both debug and release builds.
/// Throws `std::runtime_error` if `expr` evaluates to `false`.
/// Prefer this over @ref DNDS_assert for user-input and API-contract checks.
#define DNDS_check_throw(expr) \
    ((static_cast<bool>(expr)) \
         ? void(0)             \
         : ::DNDS::assert_false_info_throw(#expr, __FILE__, __LINE__, ""))

/// @brief Same as @ref DNDS_check_throw but attaches a user-supplied `info` message
/// to the thrown `std::runtime_error`.
#define DNDS_check_throw_info(expr, info) \
    ((static_cast<bool>(expr))            \
         ? void(0)                        \
         : ::DNDS::assert_false_info_throw(#expr, __FILE__, __LINE__, info))

/// Maximum assertion level — assertions at this level are ALWAYS compiled in,
/// regardless of DNDS_NDEBUG or DNDS_ASSERT_LEVEL settings.
#define DNDS_ASSERT_LEVEL_MAX 3

/// Assertion threshold: assertions with level < DNDS_ASSERT_LEVEL (and level < MAX)
/// are compiled out. Default: 0 (all active) in debug, MAX+1 (only MAX active) under NDEBUG.
/// Override at compile time with -DDNDS_ASSERT_LEVEL=N to keep levels N..MAX active.
#ifndef DNDS_ASSERT_LEVEL
#    ifdef DNDS_NDEBUG
#        define DNDS_ASSERT_LEVEL (DNDS_ASSERT_LEVEL_MAX + 1)
#    else
#        define DNDS_ASSERT_LEVEL 0
#    endif
#endif

// ---- Two-level token-paste helper (needed so level expands before pasting) ----

#define DNDS__CAT_I(a, b) a##b
#define DNDS__CAT(a, b) DNDS__CAT_I(a, b)

// ---- Per-level preprocessor dispatch (levels 0..DNDS_ASSERT_LEVEL_MAX) ----
//
// For each level L, three inner macros are `#define`d to either the active
// assertion body or `(void)(0)`. The public `_l` macros token-paste into
// the corresponding inner macro so the entire decision is resolved by the
// preprocessor — no compiler optimisations required.

/// @name Level-0 macros (legacy default verbosity)
/// @{

#if !defined(DNDS_NDEBUG) && (0 >= DNDS_ASSERT_LEVEL || 0 >= DNDS_ASSERT_LEVEL_MAX)
#    define DNDS__ASSERT_L0(expr) ((static_cast<bool>(expr)) ? void(0) : ::DNDS::assert_false(#expr, __FILE__, __LINE__))
#    define DNDS__ASSERT_INFO_L0(expr, info) \
        ((static_cast<bool>(expr)) ? void(0) : ::DNDS::assert_false_info(#expr, __FILE__, __LINE__, info))
#    define DNDS__ASSERT_INFOF_L0(expr, info, ...) \
        ((static_cast<bool>(expr)) ? void(0) : ::DNDS::assert_false_infof(#expr, __FILE__, __LINE__, info, ##__VA_ARGS__))
#else
#    define DNDS__ASSERT_L0(expr) (void(0))
#    define DNDS__ASSERT_INFO_L0(expr, info) (void(0))
#    define DNDS__ASSERT_INFOF_L0(expr, info, ...) (void(0))
#endif
/// @}

#if !defined(DNDS_NDEBUG) && (1 >= DNDS_ASSERT_LEVEL || 1 >= DNDS_ASSERT_LEVEL_MAX)
#    define DNDS__ASSERT_L1(expr) ((static_cast<bool>(expr)) ? void(0) : ::DNDS::assert_false(#expr, __FILE__, __LINE__))
#    define DNDS__ASSERT_INFO_L1(expr, info) \
        ((static_cast<bool>(expr)) ? void(0) : ::DNDS::assert_false_info(#expr, __FILE__, __LINE__, info))
#    define DNDS__ASSERT_INFOF_L1(expr, info, ...) \
        ((static_cast<bool>(expr)) ? void(0) : ::DNDS::assert_false_infof(#expr, __FILE__, __LINE__, info, ##__VA_ARGS__))
#else
#    define DNDS__ASSERT_L1(expr) (void(0))
#    define DNDS__ASSERT_INFO_L1(expr, info) (void(0))
#    define DNDS__ASSERT_INFOF_L1(expr, info, ...) (void(0))
#endif

#if !defined(DNDS_NDEBUG) && (2 >= DNDS_ASSERT_LEVEL || 2 >= DNDS_ASSERT_LEVEL_MAX)
#    define DNDS__ASSERT_L2(expr) ((static_cast<bool>(expr)) ? void(0) : ::DNDS::assert_false(#expr, __FILE__, __LINE__))
#    define DNDS__ASSERT_INFO_L2(expr, info) \
        ((static_cast<bool>(expr)) ? void(0) : ::DNDS::assert_false_info(#expr, __FILE__, __LINE__, info))
#    define DNDS__ASSERT_INFOF_L2(expr, info, ...) \
        ((static_cast<bool>(expr)) ? void(0) : ::DNDS::assert_false_infof(#expr, __FILE__, __LINE__, info, ##__VA_ARGS__))
#else
#    define DNDS__ASSERT_L2(expr) (void(0))
#    define DNDS__ASSERT_INFO_L2(expr, info) (void(0))
#    define DNDS__ASSERT_INFOF_L2(expr, info, ...) (void(0))
#endif

#if !defined(DNDS_NDEBUG) && (3 >= DNDS_ASSERT_LEVEL || 3 >= DNDS_ASSERT_LEVEL_MAX)
#    define DNDS__ASSERT_L3(expr) ((static_cast<bool>(expr)) ? void(0) : ::DNDS::assert_false(#expr, __FILE__, __LINE__))
#    define DNDS__ASSERT_INFO_L3(expr, info) \
        ((static_cast<bool>(expr)) ? void(0) : ::DNDS::assert_false_info(#expr, __FILE__, __LINE__, info))
#    define DNDS__ASSERT_INFOF_L3(expr, info, ...) \
        ((static_cast<bool>(expr)) ? void(0) : ::DNDS::assert_false_infof(#expr, __FILE__, __LINE__, info, ##__VA_ARGS__))
#else
#    define DNDS__ASSERT_L3(expr) (void(0))
#    define DNDS__ASSERT_INFO_L3(expr, info) (void(0))
#    define DNDS__ASSERT_INFOF_L3(expr, info, ...) (void(0))
#endif

// ---- Public leveled assertion macros ----
//
// Example usage: DNDS_assert_l(3, ptr != nullptr);
// Token-pastes the level digit into the corresponding inner macro above,
// which the preprocessor has already resolved to either an active body
// or (void)(0).

/// @brief Leveled assertion. Active when `level >= DNDS_ASSERT_LEVEL` or
/// `level >= DNDS_ASSERT_LEVEL_MAX`. Otherwise preprocessor-resolved to `(void)(0)`.
#define DNDS_assert_l(level, expr) DNDS__CAT(DNDS__ASSERT_L, level)(expr)
/// @brief Leveled assertion with info message.
#define DNDS_assert_info_l(level, expr, info) DNDS__CAT(DNDS__ASSERT_INFO_L, level)(expr, info)
/// @brief Leveled assertion with printf-style format message.
#define DNDS_assert_infof_l(level, expr, info, ...) \
    DNDS__CAT(DNDS__ASSERT_INFOF_L, level)          \
    (expr, info, ##__VA_ARGS__)

/// @brief MAX-level assertion — compiled in unless DNDS_NDEBUG is defined.
/// DNDS_ASSERT_LEVEL cannot disable this level. Equivalent to @ref DNDS_assert_l(DNDS_ASSERT_LEVEL_MAX, expr).
#define DNDS_assert(expr) DNDS_assert_l(DNDS_ASSERT_LEVEL_MAX, expr)
/// @brief MAX-level assertion with an extra std::string `info` message.
#define DNDS_assert_info(expr, info) DNDS_assert_info_l(DNDS_ASSERT_LEVEL_MAX, expr, info)
/// @brief MAX-level assertion with a printf-style format message.
#define DNDS_assert_infof(expr, info, ...) DNDS_assert_infof_l(DNDS_ASSERT_LEVEL_MAX, expr, info, ##__VA_ARGS__)

// ---- Public HD leveled assertion macros ----
//
// Token-paste dispatch works on both host and device — the inner macros
// DNDS__HD_ASSERT_L{level} are defined separately in each path below.

/// @brief Leveled host/device assertion. Abort on host, PTX `trap` on CUDA device.
#define DNDS_HD_assert_l(level, cond) DNDS__CAT(DNDS__HD_ASSERT_L, level)(cond)
/// @brief Leveled host/device assertion with printf-format message.
#define DNDS_HD_assert_infof_l(level, cond, info, ...) \
    DNDS__CAT(DNDS__HD_ASSERT_INFOF_L, level)          \
    (cond, info, ##__VA_ARGS__)

/// @brief MAX-level host/device assertion — always compiled unless
/// DNDS_NDEBUG (host) or DNDS_NDEBUG_DEVICE (device) override.
#define DNDS_HD_assert(cond) DNDS_HD_assert_l(DNDS_ASSERT_LEVEL_MAX, cond)
/// @brief MAX-level host/device assertion with printf-format message.
#define DNDS_HD_assert_infof(cond, info, ...) \
    DNDS_HD_assert_infof_l(DNDS_ASSERT_LEVEL_MAX, cond, info, ##__VA_ARGS__)

#ifdef __CUDA_ARCH__

/// @brief Device-side assertion failure: print once (atomic-guarded) and trap.
/// @details Uses `atomicCAS` on a managed flag so only the first failing thread
/// prints; all threads trap after the guarded block to ensure immediate
/// kernel termination. Avoids flooding the console when a kernel has one
/// bug hit by thousands of threads.
__device__ inline void device_assert_fail(const char *expr, const char *file, int line)
{
    __device__ __managed__ static int g_assert_printed = 0;
    if (atomicCAS(&g_assert_printed, 0, 1) == 0)
    {
        printf("Device assert failed: %s at %s:%d (block %d thread %d)\n",
               expr, file, line, blockIdx.x, threadIdx.x);
    }
    asm("trap;"); // force termination — all threads trap
}

/// @brief Printf-formatted variant of #device_assert_fail.
__device__ inline void device_assert_fail_infof(const char *expr, const char *file, int line,
                                                const char *info, ...)
{
    __device__ __managed__ static int g_assert_printed = 0;
    if (atomicCAS(&g_assert_printed, 0, 1) == 0)
    {
        printf("Device assert failed: %s at %s:%d (block %d thread %d)\n",
               expr, file, line, blockIdx.x, threadIdx.x);
        printf("%s\n", info);
    }
    asm("trap;"); // force termination — all threads trap
}

// ---- Per-level HD inner macros for CUDA device ----

#    if !defined(DNDS_NDEBUG) && !defined(DNDS_NDEBUG_DEVICE) && (0 >= DNDS_ASSERT_LEVEL || 0 >= DNDS_ASSERT_LEVEL_MAX)
#        define DNDS__HD_ASSERT_L0(cond)                           \
            do                                                     \
            {                                                      \
                if (!(cond))                                       \
                {                                                  \
                    device_assert_fail(#cond, __FILE__, __LINE__); \
                }                                                  \
            } while (0)
#        define DNDS__HD_ASSERT_INFOF_L0(cond, info, ...)                                             \
            do                                                                                        \
            {                                                                                         \
                if (!(cond))                                                                          \
                {                                                                                     \
                    device_assert_fail_infof(#cond, __FILE__, __LINE__, (char *)info, ##__VA_ARGS__); \
                }                                                                                     \
            } while (0)
#    else
#        define DNDS__HD_ASSERT_L0(cond) (void(0))
#        define DNDS__HD_ASSERT_INFOF_L0(cond, info, ...) (void(0))
#    endif

#    if !defined(DNDS_NDEBUG) && !defined(DNDS_NDEBUG_DEVICE) && (1 >= DNDS_ASSERT_LEVEL || 1 >= DNDS_ASSERT_LEVEL_MAX)
#        define DNDS__HD_ASSERT_L1(cond)                           \
            do                                                     \
            {                                                      \
                if (!(cond))                                       \
                {                                                  \
                    device_assert_fail(#cond, __FILE__, __LINE__); \
                }                                                  \
            } while (0)
#        define DNDS__HD_ASSERT_INFOF_L1(cond, info, ...)                                             \
            do                                                                                        \
            {                                                                                         \
                if (!(cond))                                                                          \
                {                                                                                     \
                    device_assert_fail_infof(#cond, __FILE__, __LINE__, (char *)info, ##__VA_ARGS__); \
                }                                                                                     \
            } while (0)
#    else
#        define DNDS__HD_ASSERT_L1(cond) (void(0))
#        define DNDS__HD_ASSERT_INFOF_L1(cond, info, ...) (void(0))
#    endif

#    if !defined(DNDS_NDEBUG) && !defined(DNDS_NDEBUG_DEVICE) && (2 >= DNDS_ASSERT_LEVEL || 2 >= DNDS_ASSERT_LEVEL_MAX)
#        define DNDS__HD_ASSERT_L2(cond)                           \
            do                                                     \
            {                                                      \
                if (!(cond))                                       \
                {                                                  \
                    device_assert_fail(#cond, __FILE__, __LINE__); \
                }                                                  \
            } while (0)
#        define DNDS__HD_ASSERT_INFOF_L2(cond, info, ...)                                             \
            do                                                                                        \
            {                                                                                         \
                if (!(cond))                                                                          \
                {                                                                                     \
                    device_assert_fail_infof(#cond, __FILE__, __LINE__, (char *)info, ##__VA_ARGS__); \
                }                                                                                     \
            } while (0)
#    else
#        define DNDS__HD_ASSERT_L2(cond) (void(0))
#        define DNDS__HD_ASSERT_INFOF_L2(cond, info, ...) (void(0))
#    endif

#    if !defined(DNDS_NDEBUG) && !defined(DNDS_NDEBUG_DEVICE) && (3 >= DNDS_ASSERT_LEVEL || 3 >= DNDS_ASSERT_LEVEL_MAX)
#        define DNDS__HD_ASSERT_L3(cond)                           \
            do                                                     \
            {                                                      \
                if (!(cond))                                       \
                {                                                  \
                    device_assert_fail(#cond, __FILE__, __LINE__); \
                }                                                  \
            } while (0)
#        define DNDS__HD_ASSERT_INFOF_L3(cond, info, ...)                                             \
            do                                                                                        \
            {                                                                                         \
                if (!(cond))                                                                          \
                {                                                                                     \
                    device_assert_fail_infof(#cond, __FILE__, __LINE__, (char *)info, ##__VA_ARGS__); \
                }                                                                                     \
            } while (0)
#    else
#        define DNDS__HD_ASSERT_L3(cond) (void(0))
#        define DNDS__HD_ASSERT_INFOF_L3(cond, info, ...) (void(0))
#    endif

#else // !__CUDA_ARCH__

// ---- Per-level HD inner macros for host ----
// On host, HD assertions delegate to regular DNDS_assert which already
// enforces the level system. DNDS_NDEBUG is handled by DNDS__ASSERT_L{level}.

#    define DNDS__HD_ASSERT_L0(cond) DNDS__ASSERT_L0(cond)
#    define DNDS__HD_ASSERT_INFOF_L0(cond, info, ...) DNDS__ASSERT_INFOF_L0(cond, info, ##__VA_ARGS__)
#    define DNDS__HD_ASSERT_L1(cond) DNDS__ASSERT_L1(cond)
#    define DNDS__HD_ASSERT_INFOF_L1(cond, info, ...) DNDS__ASSERT_INFOF_L1(cond, info, ##__VA_ARGS__)
#    define DNDS__HD_ASSERT_L2(cond) DNDS__ASSERT_L2(cond)
#    define DNDS__HD_ASSERT_INFOF_L2(cond, info, ...) DNDS__ASSERT_INFOF_L2(cond, info, ##__VA_ARGS__)
#    define DNDS__HD_ASSERT_L3(cond) DNDS__ASSERT_L3(cond)
#    define DNDS__HD_ASSERT_INFOF_L3(cond, info, ...) DNDS__ASSERT_INFOF_L3(cond, info, ##__VA_ARGS__)

#endif

#ifdef __CUDA_ARCH__
#    ifdef DNDS_DEVICE_BAN_EIGEN_MALLOC_DYNAMIC
#        define EIGEN_RUNTIME_NO_MALLOC
#    else
#        define EIGEN_NO_MALLOC
#    endif
#    define eigen_assert(expr) DNDS_HD_assert(expr) //! we overwrite the eigen's assert
#    if defined(EIGEN_RUNTIME_NO_MALLOC) && !defined(EIGEN_NO_MALLOC)
#        define DNDS_DEVICE_CODE_GUARD_EIGEN_MALLOC (Eigen::internal::set_is_malloc_allowed(false))
#    endif

#else

#    define DNDS_DEVICE_CODE_GUARD_EIGEN_MALLOC (void(0))

#endif

namespace DNDS
{
}
