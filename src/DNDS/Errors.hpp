#pragma once
/// @file Errors.hpp
/// @brief Assertion / error-handling macros and supporting helper functions.
///
/// ## Overview
/// Three distinct families of checks are provided; choose based on how the
/// failure should surface:
///
/// | Macro                  | Release behaviour       | Failure mode              |
/// |------------------------|-------------------------|---------------------------|
/// | @ref DNDS_assert            | Compiled out (NDEBUG)   | `std::abort()`            |
/// | @ref DNDS_assert_info       | Compiled out (NDEBUG)   | `std::abort()` + message  |
/// | @ref DNDS_assert_infof      | Compiled out (NDEBUG)   | `std::abort()` + fmtprintf|
/// | @ref DNDS_check_throw       | Always active           | `throw std::runtime_error`|
/// | @ref DNDS_check_throw_info  | Always active           | `throw` + message         |
/// | @ref DNDS_HD_assert         | Compiled out in NDEBUG  | host: `abort`, device: `trap` |
///
/// Prefer @ref DNDS_assert for internal invariants that are expensive to check or
/// cannot fail in correct code; use @ref DNDS_check_throw for user-input / runtime
/// validation that must remain active in release builds.
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

#ifdef DNDS_NDEBUG
#    define DNDS_assert(expr) (void(0))
#    define DNDS_assert_info(expr, info) (void(0))
#    define DNDS_assert_infof(expr, info, ...) (void(0))
#else
/// @brief Debug-only assertion (compiled out when @ref DNDS_NDEBUG is defined).
/// Prints the expression + file/line + backtrace, then calls `std::abort()`.
#    define DNDS_assert(expr)      \
        ((static_cast<bool>(expr)) \
             ? void(0)             \
             : ::DNDS::assert_false(#expr, __FILE__, __LINE__))
/// @brief Debug-only assertion with an extra std::string `info` message.
#    define DNDS_assert_info(expr, info) \
        ((static_cast<bool>(expr))       \
             ? void(0)                   \
             : ::DNDS::assert_false_info(#expr, __FILE__, __LINE__, info))
/// @brief Debug-only assertion with a printf-style format message.
#    define DNDS_assert_infof(expr, info, ...) \
        ((static_cast<bool>(expr))             \
             ? void(0)                         \
             : ::DNDS::assert_false_infof(#expr, __FILE__, __LINE__, info, ##__VA_ARGS__))
#endif

#ifdef __CUDA_ARCH__

/// @brief Device-side assertion failure: print once (atomic-guarded) and trap.
/// @details Uses `atomicCAS` on a managed flag so only the first failing thread
/// prints; all other threads simply call `asm("trap;")`. Avoids flooding the
/// console when a kernel has one bug hit by thousands of threads.
__device__ inline void device_assert_fail(const char *expr, const char *file, int line)
{
    __device__ __managed__ static int g_assert_printed = 0;
    if (atomicCAS(&g_assert_printed, 0, 1) == 0)
    {
        printf("Device assert failed: %s at %s:%d (block %d thread %d)\n",
               expr, file, line, blockIdx.x, threadIdx.x);
        asm("trap;"); // force termination
    }
}

/// @brief Printf-formatted variant of #device_assert_fail.
__device__ inline void device_assert_fail_infof(const char *expr, const char *file, int line,
                                                char *info, ...)
{
    __device__ __managed__ static int g_assert_printed = 0;
    if (atomicCAS(&g_assert_printed, 0, 1) == 0)
    {
        va_list args;
        va_start(args, info);
        printf("Device assert failed: %s at %s:%d (block %d thread %d)\n",
               expr, file, line, blockIdx.x, threadIdx.x);
        vprintf(info, args);
        va_end(args);
        asm("trap;"); // force termination
    }
}

#    if defined(DNDS_NDEBUG) || defined(DNDS_NDEBUG_DEVICE)
#        define DNDS_HD_assert(cond) (void(0))
#        define DNDS_HD_assert_infof(cond, info, ...) (void(0))
#    else
/// @brief Host/device assertion: abort on host, PTX `trap` on CUDA device.
/// @details Can be used inside `__host__ __device__` functions. Disabled when
/// either @ref DNDS_NDEBUG (host+device) or @ref DNDS_NDEBUG_DEVICE (device-only) is set.
#        define DNDS_HD_assert(cond)                               \
            do                                                     \
            {                                                      \
                if (!(cond))                                       \
                {                                                  \
                    device_assert_fail(#cond, __FILE__, __LINE__); \
                }                                                  \
            } while (0)

/// @brief Host/device assertion with a printf-format message.
#        define DNDS_HD_assert_infof(cond, info, ...)              \
            do                                                     \
            {                                                      \
                if (!(cond))                                       \
                {                                                  \
                    device_assert_fail(#cond, __FILE__, __LINE__); \
                }                                                  \
            } while (0)

#    endif
#else

// HOST version
/// @brief Host-only expansion of @ref DNDS_HD_assert (equivalent to @ref DNDS_assert).
#    define DNDS_HD_assert(cond) DNDS_assert(cond)
/// @brief Host-only expansion of @ref DNDS_HD_assert_infof.
#    define DNDS_HD_assert_infof(cond, info, ...) DNDS_assert_infof(cond, info, ##__VA_ARGS__)
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