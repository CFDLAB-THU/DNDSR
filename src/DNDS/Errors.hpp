#pragma once

#include "Defines.hpp"

#ifdef __CUDA_ARCH__
__device__ inline void device_assert_fail(const char *expr, const char *file, int line)
{
    printf("Device assert failed: %s at %s:%d (block %d thread %d)\n",
           expr, file, line, blockIdx.x, threadIdx.x);
    asm("trap;"); // force termination of this thread (optional)
}

__device__ inline void device_assert_fail_infof(const char *expr, const char *file, int line,
                                               char *info, ...)
{
    va_list args;
    va_start(args, info);
    printf("Device assert failed: %s at %s:%d (block %d thread %d)\n",
           expr, file, line, blockIdx.x, threadIdx.x);
    printf(info, args);
    asm("trap;"); // force termination of this thread (optional)
}

#define DNDS_HD_assert(cond)                               \
    do                                                     \
    {                                                      \
        if (!(cond))                                       \
        {                                                  \
            device_assert_fail(#cond, __FILE__, __LINE__); \
        }                                                  \
    } while (0)

#define DNDS_HD_assert_infof(cond, info, ...)              \
    do                                                     \
    {                                                      \
        if (!(cond))                                       \
        {                                                  \
            device_assert_fail(#cond, __FILE__, __LINE__); \
        }                                                  \
    } while (0)
#else

// HOST version
#define DNDS_HD_assert(cond) DNDS_assert(cond)
#define DNDS_HD_assert_infof(cond, info, ...) DNDS_assert_infof(cond, info, ##__VA_ARGS__)
#endif

namespace DNDS
{
}