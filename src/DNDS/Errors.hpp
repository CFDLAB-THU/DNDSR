#pragma once

#include "Macros.hpp"

// assert macros

#include <iostream>
#include <cstdarg>

std::string __DNDS_getTraceString();

inline void __DNDS_assert_false(const char *expr, const char *file, int line)
{
    std::cerr << __DNDS_getTraceString() << "\n";
    std::cerr << "\033[91m DNDS_assertion failed\033[39m: \"" << expr << "\"  at [  " << file << ":" << line << "  ]" << std::endl;
    std::abort();
}

inline void __DNDS_assert_false_info(const char *expr, const char *file, int line, const std::string &info)
{
    std::cerr << __DNDS_getTraceString() << "\n";
    std::cerr << "\033[91m DNDS_assertion failed\033[39m: \"" << expr << "\"  at [  " << file << ":" << line << "  ]\n"
              << info << std::endl;
    std::abort();
}

inline void __DNDS_assert_false_infof(const char *expr, const char *file, int line,
                                      const char *info, ...)
{
    va_list args;
    va_start(args, info);
    std::cerr << __DNDS_getTraceString() << "\n";
    std::cerr << "\033[91m DNDS_assertion failed\033[39m: \"" << expr << "\"  at [  " << file << ":" << line << "  ]\n";
    char format_buf[1024 * 512];
    std::vsnprintf(format_buf, sizeof(format_buf), info, args);
    va_end(args);
    std::cerr << format_buf << std::endl;
    std::abort();
}

#ifdef DNDS_NDEBUG
#    define DNDS_assert(expr) (void(0))
#    define DNDS_assert_info(expr, info) (void(0))
#    define DNDS_assert_infof(expr, info, ...) (void(0))
#else
#    define DNDS_assert(expr)      \
        ((static_cast<bool>(expr)) \
             ? void(0)             \
             : __DNDS_assert_false(#expr, __FILE__, __LINE__))
#    define DNDS_assert_info(expr, info) \
        ((static_cast<bool>(expr))       \
             ? void(0)                   \
             : __DNDS_assert_false_info(#expr, __FILE__, __LINE__, info))
#    define DNDS_assert_infof(expr, info, ...) \
        ((static_cast<bool>(expr))             \
             ? void(0)                         \
             : __DNDS_assert_false_infof(#expr, __FILE__, __LINE__, info, ##__VA_ARGS__))
#endif

#ifdef __CUDA_ARCH__

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
#        define DNDS_HD_assert(cond)                               \
            do                                                     \
            {                                                      \
                if (!(cond))                                       \
                {                                                  \
                    device_assert_fail(#cond, __FILE__, __LINE__); \
                }                                                  \
            } while (0)

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
#    define DNDS_HD_assert(cond) DNDS_assert(cond)
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