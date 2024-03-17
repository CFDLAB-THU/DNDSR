#pragma once

/*------------------------------------------*/
// Warning disabler:

#if defined(_MSC_VER) && defined(_WIN32) && !defined(__clang__)
#define DISABLE_WARNING_PUSH __pragma(warning(push))
#define DISABLE_WARNING_POP __pragma(warning(pop))
#define DISABLE_WARNING(warningNumber) __pragma(warning(disable \
                                                        : warningNumber))

#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER DISABLE_WARNING(4100)
#define DISABLE_WARNING_UNREFERENCED_FUNCTION DISABLE_WARNING(4505)
#define DISABLE_WARNING_DEPRECATED_DECLARATIONS
#define DISABLE_WARNING_UNUSED_VALUE
#define DISABLE_WARNING_MAYBE_UNINITIALIZED
// other warnings you want to deactivate...

#elif defined(_MSC_VER) && defined(_WIN32) && defined(__clang__) // for clang-msvc on win, change a bit from unix version

#define DO_PRAGMA(X) _Pragma(#X)
#define DISABLE_WARNING_PUSH DO_PRAGMA(GCC diagnostic push)
#define DISABLE_WARNING_POP DO_PRAGMA(GCC diagnostic pop)
#define DISABLE_WARNING(warningName) DO_PRAGMA(GCC diagnostic ignored warningName)

#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER DISABLE_WARNING("-Wunused-parameter")
#define DISABLE_WARNING_UNREFERENCED_FUNCTION DISABLE_WARNING("-Wunused-function")
#define DISABLE_WARNING_DEPRECATED_DECLARATIONS DISABLE_WARNING("-Wdeprecated-declarations")
#define DISABLE_WARNING_UNUSED_VALUE DISABLE_WARNING("-Wunused-value")
#define DISABLE_WARNING_MAYBE_UNINITIALIZED 

#elif defined(__GNUC__) || defined(__clang__) // unix + gcc/clang
#define DO_PRAGMA(X) _Pragma(#X)
#define DISABLE_WARNING_PUSH DO_PRAGMA(GCC diagnostic push)
#define DISABLE_WARNING_POP DO_PRAGMA(GCC diagnostic pop)
#define DISABLE_WARNING(warningName) DO_PRAGMA(GCC diagnostic ignored warningName)

#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER DISABLE_WARNING("-Wunused-parameter")
#define DISABLE_WARNING_UNREFERENCED_FUNCTION DISABLE_WARNING("-Wunused-function")
#define DISABLE_WARNING_DEPRECATED_DECLARATIONS DISABLE_WARNING("-Wdeprecated-declarations")
#define DISABLE_WARNING_UNUSED_VALUE DISABLE_WARNING("-Wunused-value")
#define DISABLE_WARNING_MAYBE_UNINITIALIZED DISABLE_WARNING("-Wmaybe-uninitialized")
// other warnings you want to deactivate...

#else
#define DISABLE_WARNING_PUSH
#define DISABLE_WARNING_POP
#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER
#define DISABLE_WARNING_UNUSED_VALUE
#define DISABLE_WARNING_UNREFERENCED_FUNCTION
// other warnings you want to deactivate...

#endif

/*------------------------------------------*/