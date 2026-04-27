#pragma once
/// @file Defines.hpp
/// @brief Core type aliases, constants, and metaprogramming utilities for the DNDS framework.

#include "Macros.hpp"
// #define NDEBUG
#include "Errors.hpp"
#include "EigenPCH.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <memory>
#include <tuple>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>
#include <type_traits>
#include <filesystem>
#include <functional>
#include <locale>
#include <csignal>
#include <cstdarg>
#include <cstdlib> // must be included before fmt headers for malloc/free with libc++ _LIBCPP_REMOVE_TRANSITIVE_INCLUDES

#include <fmt/core.h>

#if defined(linux) || defined(_UNIX) || defined(__linux__) || defined(__unix__) || defined(__APPLE__)
#    define DNDS_UNIX_LIKE
#endif

#ifdef DNDS_USE_OMP
#    include <omp.h>
#endif

#if defined(_WIN32) || defined(__WINDOWS_)
// #include <Windows.h>
// #include <process.h>
#endif

#define DNDS_FMT_ARG(V) fmt::arg(#V, V)

/***************/ // DNDS_assertS

extern "C" void DNDS_signal_handler(int signal);

namespace DNDS
{
    /// @brief Install SEGV / ABRT handlers that print a backtrace via @ref DNDS_signal_handler.
    /// @details Optional opt-in; not installed automatically so unit test drivers
    /// keep full control of their own handlers. Typical drivers call this once
    /// after `MPI_Init` in `main()`.
    inline void RegisterSignalHandler()
    {
        std::signal(SIGSEGV, DNDS_signal_handler);
        std::signal(SIGABRT, DNDS_signal_handler);
        // std::signal(SIGKILL, DNDS_signal_handler);
    }
}

/***************/

#ifdef DNDS_USE_CUDA
#    include <cuda_runtime.h>
#endif

#if defined(DNDS_USE_CUDA)
#    define DNDS_DEVICE_CALLABLE __host__ __device__
#    define DNDS_DEVICE __device__
#    define DNDS_HOST __host__
#    define DNDS_GLOBAL __global__
#    define DNDS_CONSTANT __constant__
#else
#    define DNDS_DEVICE_CALLABLE
#    define DNDS_DEVICE
#    define DNDS_HOST
#    define DNDS_GLOBAL
#    define DNDS_CONSTANT
#endif

#define DNDS_DEVICE_TRIVIAL_COPY_DEFINE(T, T_Self)               \
    DNDS_DEVICE_CALLABLE T() = default;                          \
    DNDS_DEVICE_CALLABLE T(const T_Self &) = default;            \
    DNDS_DEVICE_CALLABLE T(T_Self &&) = default;                 \
    DNDS_DEVICE_CALLABLE T &operator=(const T_Self &) = default; \
    DNDS_DEVICE_CALLABLE T &operator=(T_Self &&) = default;      \
    DNDS_DEVICE_CALLABLE ~T() = default;

#define DNDS_DEVICE_TRIVIAL_COPY_DEFINE_NO_EMPTY_CTOR(T, T_Self) \
    DNDS_DEVICE_CALLABLE T(const T_Self &) = default;            \
    DNDS_DEVICE_CALLABLE T(T_Self &&) = default;                 \
    DNDS_DEVICE_CALLABLE T &operator=(const T_Self &) = default; \
    DNDS_DEVICE_CALLABLE T &operator=(T_Self &&) = default;      \
    DNDS_DEVICE_CALLABLE ~T() = default;

/***************/

static_assert(sizeof(uint8_t) == 1, "bad uint8_t");

namespace DNDS
{
    /// @brief Canonical floating-point scalar used throughout DNDSR (double precision).
    using real = double;
    /// @brief Global row / DOF index type (signed 64-bit; handles multi-billion-cell meshes).
    using index = int64_t;
    /// @brief Row-width / per-row element-count type (signed 32-bit).
    using rowsize = int32_t;
    /// @brief Integer type with the same width as #real (used for type-punning / packing).
    using real_sized_index = int64_t;
    /// @brief Integer type with half the width of #real.
    using real_half_sized_index = int32_t;
    static_assert(sizeof(real_sized_index) == sizeof(real) && sizeof(real_half_sized_index) == sizeof(real) / 2);

#define DNDS_INDEX_MAX INT64_MAX
#define DNDS_INDEX_MIN INT64_MIN
#define DNDS_ROWSIZE_MAX INT32_MAX
#define DNDS_ROWSIZE_MIN INT32_MIN

    /// @brief Default column separator for CLI / log output (tab character).
    static const char *outputDelim = "\t";

    /// @brief Split a global range `[0, nGlobal)` evenly among `nRanks` workers.
    /// @details Used by the @ref EvenSplit serialization-offset mode and by
    /// redistribution routines to pick per-rank slabs without exchanging sizes.
    /// The partition is dense and balanced to within one item.
    /// @return `[start, end)` for the caller's `rank`.
    inline std::pair<index, index> EvenSplitRange(int rank, int nRanks, index nGlobal)
    {
        index start = index(rank) * nGlobal / index(nRanks);
        index end = index(rank + 1) * nGlobal / index(nRanks);
        return {start, end};
    }

    /// @brief Shortened alias for `std::shared_ptr` used pervasively in DNDSR.
    template <typename T>
    using ssp = std::shared_ptr<T>;

    /// @brief Type trait that detects whether a type is a std::shared_ptr wrapping.
    template <typename T>
    struct is_ssp : std::false_type
    {
    };

    template <typename T>
    struct is_ssp<ssp<T>> : std::true_type
    {
    };

    template <typename T>
    inline constexpr bool is_ssp_v = is_ssp<T>::value;

    /// @brief Convenience `remove_cv` + `remove_reference` composition (C++17 port of C++20's `std::remove_cvref_t`).
    template <typename T>
    using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

    /// @brief Vector of row widths (one `rowsize` per row).
    using t_RowsizeVec = std::vector<rowsize>;
    /// @brief Vector of index values (global offsets, local ids, etc.).
    using t_IndexVec = std::vector<index>;
    /// @brief Shared pointer alias to t_IndexVec (used by mapping tables shared
    /// between arrays, see IndexMapping.hpp).
    using t_pIndexVec = ssp<t_IndexVec>;

    /// @brief Paired indices, typically `(start, size)` or `(first, last)`.
    using t_indexerPair = std::tuple<index, index>;

    /// @brief Minimum representable #index value (= @ref INT64_MIN).
    DNDS_CONSTANT const index indexMin = INT64_MIN;

    /// @brief Sentinel "not initialised" real value (NaN). Cheap to detect with
    /// `std::isnan` or @ref IsUnInitReal; survives MPI transport unchanged.
    DNDS_CONSTANT const real UnInitReal = NAN;
    /// @brief Sentinel "not initialised" index value (= @ref INT64_MIN).
    DNDS_CONSTANT const index UnInitIndex = INT64_MIN;
    static_assert(UnInitIndex < 0);
    /// @brief Sentinel "not initialised" rowsize value (= @ref INT32_MIN).
    DNDS_CONSTANT const rowsize UnInitRowsize = INT32_MIN;
    static_assert(UnInitRowsize < 0);

    /// @brief Whether `v` equals the NaN sentinel @ref UnInitReal (tested via `isnan`).
    inline bool IsUnInitReal(real v)
    {
        // return (*(int64_t *)(&v)) == (*(int64_t *)(&UnInitReal));
        return std::isnan(v);
    }

    /// @brief Catch-all upper bound ("practically infinity") for physical scalars.
    DNDS_CONSTANT const real veryLargeReal = 3e200;
    /// @brief Loose upper bound (e.g., for non-dimensional limits).
    DNDS_CONSTANT const real largeReal = 3e10;
    /// @brief Catch-all lower bound ("effectively zero").
    DNDS_CONSTANT const real verySmallReal = 1e-200;
    /// @brief Loose lower bound (for iterative-solver tolerances etc.).
    DNDS_CONSTANT const real smallReal = 1e-10;
#define DNDS_E_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062
    /// @brief π in double precision (matches @ref DNDS_E_PI macro).
    DNDS_CONSTANT const real pi = DNDS_E_PI;

    /// @brief Row-major dynamic Eigen matrix used by quadrature / basis tables.
    using tDiFj = Eigen::Matrix<real, -1, -1, Eigen::RowMajor>;

    /// @brief Column-major dynamic Eigen matrix of reals (default layout).
    using MatrixXR = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
    /// @brief Dynamic Eigen vector of reals.
    using VectorXR = Eigen::Vector<real, Eigen::Dynamic>;
    /// @brief Dynamic row-vector of reals.
    using RowVectorXR = Eigen::RowVector<real, Eigen::Dynamic>;

/// @deprecated Use make_ssp<T>() or make_ssp<T>(ObjName{...}, ...) instead.
#define DNDS_MAKE_SSP(ssp, ...) (ssp = std::make_shared<typename decltype(ssp)::element_type>(__VA_ARGS__))

    /// @brief Mixin base class providing a runtime instance name for tracing/debugging.
    ///
    /// Array and its subclasses inherit this to carry a human-readable name
    /// (e.g., "coords", "cell2node") that appears in assertion messages.
    /// Zero overhead when no name is set (empty string).
    /// NOT added to device-callable types (ArrayView, ArrayLayout).
    class ObjectNaming
    {
        std::string _objectName;

    public:
        ObjectNaming() = default;
        ObjectNaming(const ObjectNaming &) = default;
        ObjectNaming(ObjectNaming &&) = default;
        ObjectNaming &operator=(const ObjectNaming &) = default;
        ObjectNaming &operator=(ObjectNaming &&) = default;
        ~ObjectNaming() = default;

        void setObjectName(const std::string &name) { _objectName = name; }
        [[nodiscard]] const std::string &getObjectName() const { return _objectName; }

        /// Returns "name(sig)" if name is set, or just "sig" otherwise.
        /// Callers pass the type signature (e.g., GetArrayName()) as `sig`.
        [[nodiscard]] std::string getObjectIdentity(const std::string &sig) const
        {
            if (_objectName.empty())
                return sig;
            return fmt::format("{}({})", _objectName, sig);
        }
    };

    /// @brief Tag type for naming objects created via make_ssp.
    ///
    /// Usage: `auto p = make_ssp<ParArray<index>>(ObjName{"cell2node"}, mpi);`
    struct ObjName
    {
        std::string name;
    };

    /// @brief Type-safe replacement for DNDS_MAKE_SSP. Creates ssp<T> with forwarded args.
    template <typename T, typename... Args>
    ssp<T> make_ssp(Args &&...args)
    {
        return std::make_shared<T>(std::forward<Args>(args)...);
    }

    /// @brief Named variant of make_ssp. If T inherits ObjectNaming, sets the name.
    ///
    /// Usage: `auto p = make_ssp<ParArray<index>>(ObjName{"cell2node"}, mpi);`
    template <typename T, typename... Args>
    ssp<T> make_ssp(ObjName objName, Args &&...args)
    {
        auto p = std::make_shared<T>(std::forward<Args>(args)...);
        if constexpr (std::is_base_of_v<ObjectNaming, T>)
            p->setObjectName(objName.name);
        return p;
    }

} // namespace DNDS

namespace DNDS
{
    /// @brief Template parameter flag: "row width is set at runtime but uniform".
    DNDS_CONSTANT const rowsize DynamicSize = -1;
    /// @brief Template parameter flag: "each row has an independent width".
    DNDS_CONSTANT const rowsize NonUniformSize = -2;
    static_assert(DynamicSize != NonUniformSize, "DynamicSize, NonUniformSize definition conflict");
    /// @brief Alignment flag: no padding applied to rows (the only currently-supported value).
    DNDS_CONSTANT const rowsize NoAlign = -1024;

    /// @brief Convert a #rowsize constant to the corresponding Eigen compile-time size.
    /// Fixed >= 0 -> the value; DynamicSize / NonUniformSize -> Eigen::Dynamic.
    constexpr int RowSize_To_EigenSize(rowsize rs)
    {
        return rs >= 0 ? static_cast<int>(rs) : (rs == DynamicSize || rs == NonUniformSize ? Eigen::Dynamic : INT_MIN);
    }

    /// @brief Encode a #rowsize constant as a short Python-binding snippet:
    /// "<number>" for fixed, "D" for DynamicSize, "I" for NonUniformSize.
    /// Used when generating pybind11 class names.
    inline std::string RowSize_To_PySnippet(rowsize rs)
    {
        if (rs >= 0)
            return fmt::format("{}", rs);
        else if (rs == DynamicSize)
            return "D";
        else if (rs == NonUniformSize)
            return "I";
        else
        {
            return "Unknown";
        }
    }

    /// @brief Encode an alignment value as a Python-binding snippet:
    /// "N" for @ref NoAlign, the number otherwise.
    inline std::string Align_To_PySnippet(rowsize al)
    {
        if (al == NoAlign)
            return "N";
        else if (al < 0)
            return "Invalid";
        else
            return fmt::format("{}", al);
    }
}

namespace DNDS
{
    /// @brief Shared output stream: where `DNDS::log()` writes progress / diagnostics.
    /// @details Defaults to `std::cout` when #useCout is true; otherwise refers
    /// to the file/stream installed via #setLogStream.
    extern ssp<std::ostream> logStream;

    /// @brief Whether `DNDS::log()` is currently routed to `std::cout`.
    extern bool useCout;

    /// @brief Return the current DNDSR log stream (either `std::cout` or the installed file).
    std::ostream &log();

    /// @brief Heuristic detection of whether `ostream` is attached to a terminal.
    bool ostreamIsTTY(std::ostream &ostream);

    /// @brief Convenience: #ostreamIsTTY applied to the current #log() stream.
    bool logIsTTY();

    /// @brief Redirect `log()` output to a user-supplied stream. Ownership is shared.
    void setLogStream(ssp<std::ostream> nstream);

    /// @brief Restore the default `std::cout` routing for `log()`.
    void setLogStreamCout();

    /// @brief Terminal width in columns (falls back to a fixed default when not a TTY).
    int get_terminal_width();

    /// @brief Render a textual progress bar to `os` for `progress` in `[0, 1]`.
    void print_progress(std::ostream &os, double progress);
}

namespace DNDS
{
    /// @brief Read `OMP_NUM_THREADS` env var, returning 1 if unset / invalid.
    extern int get_env_OMP_NUM_THREADS();
    /// @brief Read the DNDSR-specific @ref DNDS_DIST_OMP_NUM_THREADS override,
    /// falling back to #get_env_OMP_NUM_THREADS.
    extern int get_env_DNDS_DIST_OMP_NUM_THREADS();
}

/*









*/

namespace DNDS
{
    /// @brief Overflow-detecting test for `value + increment` on signed integers.
    /// @return `true` if the addition would overflow.
    // Generic function to check for overflow in signed integer addition
    template <typename T>
    bool signedIntWillAddOverflow(T value, T increment)
    {
        static_assert(std::is_signed_v<T> && std::is_integral_v<T>, "T must be a signed integral type");
        if (increment > 0 && value > std::numeric_limits<T>::max() - increment)
            return true; // Overflow when adding a positive increment
        if (increment < 0 && value < std::numeric_limits<T>::min() - increment)
            return true; // Overflow when adding a negative increment
        return false;    // No overflow
    }

    /// @brief Add two signed integers, asserting on overflow instead of silently wrapping.
    /// @throws Dies via @ref DNDS_assert_info if the result would overflow.
    // Generic function to safely add with overflow check
    template <typename T>
    T signedIntSafeAdd(T value, T increment)
    {
        static_assert(std::is_signed_v<T> && std::is_integral_v<T>, "T must be a signed integral type");
        if (signedIntWillAddOverflow<T>(value, increment))
        {
            DNDS_assert_info(false, fmt::format("doing [{} + {}] overflow detected", value, increment));
            return 0;
        }
        return value + increment;
    }

    /// @brief Narrowing `size_t -> T` conversion with range check.
    /// @tparam T  Signed integral target type.
    /// @throws Dies via @ref DNDS_assert_info if `v > std::numeric_limits<T>::max()`.
    template <typename T>
    T size_t_to_signed(size_t v)
    {
        static_assert(std::is_signed_v<T> && std::is_integral_v<T>, "T must be a signed integral type");
        static_assert(std::is_unsigned_v<size_t>, "size_t should be unsigned");
        if (v <= std::numeric_limits<T>::max())
            return T(v);
        else
        {
            DNDS_assert_info(false, fmt::format("[{}] from {} to size_t overflow detected", v, typeid(T).name()));
            return v;
        }
    }

    /// @brief Range-checked conversion from size_t to DNDS::index.
    /// @throws std::runtime_error (via DNDS_assert_info) if value exceeds index max.
    /// Use this for safe conversions from container sizes (e.g., std::vector::size())
    /// to DNDS index types, avoiding sign comparison warnings and overflow bugs.
    inline index size_to_index(size_t v)
    {
        return size_t_to_signed<index>(v);
    }

    /// @brief Range-checked conversion from size_t to DNDS::rowsize.
    /// @throws std::runtime_error (via DNDS_assert_info) if value exceeds rowsize max.
    inline rowsize size_to_rowsize(size_t v)
    {
        return size_t_to_signed<rowsize>(v);
    }

    /**
     * @brief Build a prefix-sum table from a row-size vector.
     *
     * @details Resizes `rowstarts` to `rowsizes.size() + 1` and fills it so that
     * `rowstarts[i] = sum_{k<i} rowsizes[k]`. Asserts on overflow.
     * Used extensively by CSR @ref DNDS::Array "Array" / ghost-indexing to turn row-count vectors
     * into offset vectors.
     *
     * @tparam TtRowsizeVec  `std::vector`-like of row sizes (e.g. `rowsize`).
     * @tparam TtIndexVec    `std::vector`-like of offsets (e.g. `index`), must
     *                       have a strictly wider range than the row-size type.
     */
    // Note that TtIndexVec being accumulated could overflow
    template <class TtRowsizeVec, class TtIndexVec>
    inline void AccumulateRowSize(const TtRowsizeVec &rowsizes, TtIndexVec &rowstarts)
    {
        static_assert(std::is_signed_v<typename TtIndexVec::value_type>, "row starts should be signed");
        static_assert(std::numeric_limits<typename TtIndexVec::value_type>::max() >=
                          std::numeric_limits<typename TtRowsizeVec::value_type>::max(),
                      "row starts should have larger upper limit");
        rowstarts.resize(rowsizes.size() + 1);
        rowstarts[0] = 0;
        for (typename TtIndexVec::size_type i = 1; i < rowstarts.size(); i++)
        {
            rowstarts[i] = signedIntSafeAdd<typename TtIndexVec::value_type>(rowstarts[i - 1], rowsizes[i - 1]);
        }
    }

    /// @brief Whether all elements of `dat` are equal; if so, stores the value into `value`.
    /// @return false if the vector is empty or contains at least two distinct values.
    template <class T>
    inline bool checkUniformVector(const std::vector<T> &dat, T &value)
    {
        if (dat.empty())
            return false;
        value = dat[0];
        for (auto i = 1; i < dat.size(); i++)
            if (dat[i] != value)
                return false;
        return true;
    }

    /// @brief Print a vector to `out` with #outputDelim between elements.
    /// @tparam TP  Optional cast type (e.g., promote `int8_t` to `int` to avoid
    /// character printing).
    template <class T, class TP = T>
    inline void PrintVec(const std::vector<T> &dat, std::ostream &out)
    {
        for (auto i = 0; i < dat.size(); i++)
            out << TP(dat[i]) << outputDelim;
    }

    /// @brief Integer ceiling division. `l` must be non-negative, `r` positive.
    /// @bug The current expression has incorrect precedence; see #divide_ceil for
    /// the corrected version. Do not use new code with `divCeil`.
    /// \brief l must be non-negative, r must be positive. integers
    template <class TL, class TR>
    constexpr auto divCeil(TL l, TR r)
    {
        return l / r + (l % r) ? 1 : 0;
    }
}

/*









*/

namespace DNDS
{
    /// @brief `a * a`, constexpr. Works for all arithmetic types.
    template <typename T>
    constexpr T sqr(const T &a)
    {
        static_assert(std::is_arithmetic_v<T>, "need arithmetic");
        return a * a;
    }

    /// @brief `a * a * a`, constexpr.
    template <typename T>
    constexpr T cube(const T &a)
    {
        static_assert(std::is_arithmetic_v<T>, "need arithmetic");
        return a * a * a;
    }

    /// @brief Signum function: +1, 0, or -1.
    constexpr real sign(real a)
    {
        return a > 0 ? 1 : (a < 0 ? -1 : 0);
    }

    /// @brief Tolerant signum: returns 0 inside `[-tol, tol]`.
    constexpr real signTol(real a, real tol)
    {
        return a > tol ? 1 : (a < -tol ? -1 : 0);
    }

    /// @brief "Signum, biased toward +1": treats 0 as positive.
    constexpr real signP(real a)
    {
        return a >= 0 ? 1 : -1;
    }

    /// @brief "Signum, biased toward -1": treats 0 as negative.
    constexpr real signM(real a)
    {
        return a <= 0 ? -1 : 1;
    }

    /// @brief Mathematical modulo that always returns a non-negative result.
    /// Unlike `%` in C++ where `(-1) % 3 == -1`, `mod(-1, 3) == 2`.
    template <typename T>
    constexpr T mod(T a, T b)
    {
        static_assert(std::is_signed<T>::value && std::is_integral<T>::value, "not legal mod type");
        T val = a % b;
        if (val < 0)
            val += b;
        return val;
    }

    /// @brief Integer ceiling division `ceil(a / b)`. Correct for all signs.
    template <typename T>
    constexpr T divide_ceil(T a, T b)
    {
        static_assert(std::is_integral<T>::value, "not legal mod type");
        return a / b + (a % b ? 1 : 0);
    }
    // const int a = divide_ceil(23, 11);

    /**
     * @brief Floating-point modulo matching Python's `%` (result has sign of `b`).
     * @param b Must be positive.
     */
    inline real float_mod(real a, real b)
    {
        return a - std::floor(a / b) * b;
    }

    /**
     * @brief Walk two ordered ranges in lockstep, calling `F` on each match.
     *
     * @details Two-pointer algorithm. Requires both ranges to be sorted in
     * ascending order; advances the smaller side at each step. Exits early
     * (returning `true`) if `F(value, pos1, pos2)` returns `true`.
     *
     * @tparam tF  Callable `bool(value, size_t pos1, size_t pos2)`.
     * @return true if the functor short-circuited, false if ranges were walked to completion.
     */
    template <class tIt1, class tIt1end, class tIt2, class tIt2end, class tF>
    bool iterateIdentical(tIt1 it1, tIt1end it1end, tIt2 it2, tIt2end it2end, tF F)
    {
        size_t it1Pos{0}, it2Pos{0};
        while (it1 != it1end && it2 != it2end)
        {
            if ((*it1) < (*it2))
                ++it1, ++it1Pos;
            else if ((*it1) > (*it2))
                ++it2, ++it2Pos;
            else if ((*it1) == (*it2))
            {
                if (F(*it1, it1Pos, it2Pos))
                    return true;
                ++it1, ++it1Pos;
                ++it2, ++it2Pos;
            }
        }
        return false;
    }

    ///@todo //TODO: overflow_assign_int64_to_32

    /// @brief Narrow #index to `int32_t` with range check; dies on overflow.
    inline int32_t checkedIndexTo32(index v)
    {
        DNDS_assert_info(!(v > static_cast<index>(INT32_MAX) || v < static_cast<index>(INT32_MIN)),
                         fmt::format("Index {} to int32 overflow", v));
        return static_cast<int32_t>(v);
    }

    /// @brief Convert a `wstring` to `string` (UTF-8 on Windows, byte-cast elsewhere).
    std::string getStringForceWString(const std::wstring &v);

    /// @brief Portable conversion of a platform-native path string to `std::string`.
    inline std::string getStringForcePath(const std::filesystem::path::string_type &v)
    {
#ifdef _WIN32
        return getStringForceWString(v);
#else
        return std::string{v};
#endif
    }

}

namespace DNDS
{
#if EIGEN_MAJOR_VERSION >= 5
    /// @brief Eigen "select every index" placeholder (compatibility wrapper across Eigen versions).
    static const auto EigenAll = Eigen::placeholders::all;
    /// @brief Eigen "last index" placeholder (compatibility wrapper across Eigen versions).
    static const auto EigenLast = Eigen::placeholders::last;
#else
    static const auto EigenAll = Eigen::all;
    static const auto EigenLast = Eigen::last;
#endif

}

/*-----------------------------------------*/
// some meta-programming utilities
namespace DNDS::Meta
{
    template <typename T>
    inline constexpr bool always_false = false;

    /// @brief Type trait that detects whether a type is a std::array.
    template <class T>
    struct is_std_array : std::false_type
    {
    };

    template <class T, size_t N>
    struct is_std_array<std::array<T, N>> : std::true_type
    {
    };

    /// @brief Type trait that detects whether a type is a std::vector.
    template <class T>
    struct is_std_vector : std::false_type
    {
    };

    template <class T, class Allocator>
    struct is_std_vector<std::vector<T, Allocator>> : std::true_type
    {
    };

    template <typename _Tp>
    inline constexpr bool is_std_array_v = is_std_array<_Tp>::value;

    static_assert(is_std_array_v<std::array<real, 5>> && (!is_std_array_v<std::vector<real>>)); // basic test

    /**
     * @brief see if the Actual valid data is in the struct scope (memcpy copyable)
     * @details
     * generally a fixed size Eigen::Matrix, but it seems std::is_trivially_copyable_v<> does not distinguish that,
     * see https://eigen.tuxfamily.org/dox/classEigen_1_1Matrix.html, ABI part
     * ```
     * static_assert(!is_fixed_data_real_eigen_matrix_v<std::array<real, 10>> &&
                     is_fixed_data_real_eigen_matrix_v<Eigen::Matrix<real, 2, 2>> &&
                     !is_fixed_data_real_eigen_matrix_v<Eigen::Matrix<real, -1, 2>> &&
                     is_fixed_data_real_eigen_matrix_v<Eigen::Vector2d> &&
                     !is_fixed_data_real_eigen_matrix_v<Eigen::Vector2f> &&
                     is_fixed_data_real_eigen_matrix_v<Eigen::Matrix<real, -1, -1, Eigen::DontAlign, 2, 2>> &&
                     !is_fixed_data_real_eigen_matrix_v<Eigen::Matrix<real, -1, -1, Eigen::DontAlign, -1, 2>> &&
                     !is_fixed_data_real_eigen_matrix_v<Eigen::MatrixXd>,
                 "is_fixed_data_real_eigen_matrix_v bad");
     * ```
     *
     * @tparam T
     */

    template <class T>
    struct is_fixed_data_real_eigen_matrix
    {
        static constexpr bool value = false;
    };

    template <class T, int M, int N, int options, int max_m, int max_n>
    struct is_fixed_data_real_eigen_matrix<Eigen::Matrix<T, M, N, options, max_m, max_n>>
    {
        static constexpr bool value = std::is_same_v<real, T> &&
                                      ((M > 0 && N > 0) ||
                                       (max_m > 0 && max_n > 0));
    };

    template <typename _Tp>
    inline constexpr bool is_fixed_data_real_eigen_matrix_v = is_fixed_data_real_eigen_matrix<_Tp>::value;

    static_assert(!is_fixed_data_real_eigen_matrix_v<std::array<real, 10>> &&
                      is_fixed_data_real_eigen_matrix_v<Eigen::Matrix<real, 2, 2>> &&
                      !is_fixed_data_real_eigen_matrix_v<Eigen::Matrix<real, -1, 2>> &&
                      is_fixed_data_real_eigen_matrix_v<Eigen::Vector2d> &&
                      !is_fixed_data_real_eigen_matrix_v<Eigen::Vector2f> &&
                      is_fixed_data_real_eigen_matrix_v<Eigen::Matrix<real, -1, -1, Eigen::DontAlign, 2, 2>> &&
                      !is_fixed_data_real_eigen_matrix_v<Eigen::Matrix<real, -1, -1, Eigen::DontAlign, -1, 2>> &&
                      !is_fixed_data_real_eigen_matrix_v<Eigen::MatrixXd>,
                  "is_fixed_data_real_eigen_matrix_v bad");

    template <typename T>
    inline constexpr bool is_eigen_dense_v = std::is_base_of_v<Eigen::DenseBase<T>, T>;

    /// @brief Type trait that detects whether a type is an Eigen::Matrix with `real` scalar type.
    template <class T>
    struct is_real_eigen_matrix
    {
        static constexpr bool value = false;
    };

    template <int M, int N, int options, int max_m, int max_n>
    struct is_real_eigen_matrix<Eigen::Matrix<real, M, N, options, max_m, max_n>>
    {
        static constexpr bool value = true;
    };

    template <class T>
    inline constexpr bool is_real_eigen_matrix_v = is_real_eigen_matrix<T>::value;

    /// @brief Type trait that detects whether std::hash is specialized for a given type.
    template <typename T, typename = void>
    struct has_std_hash : std::false_type
    {
    };

    template <typename T>
    struct has_std_hash<T, std::void_t<decltype(std::hash<T>{}(std::declval<T>()))>> : std::true_type
    {
    };

    static_assert(has_std_hash<double>::value);
    static_assert(!has_std_hash<std::vector<std::string>>::value);
}

namespace DNDS
{
    inline std::vector<std::string> splitSString(const std::string &str, char delim) // TODO: make more C++
    {
        std::vector<std::string> ret;
        size_t top = 0;
        size_t bot = 0;
        while (top < str.size() + 1)
        {
            if (str[top] != delim && top != str.size())
            {
                top++;
                continue;
            }
            ret.push_back(str.substr(bot, top - bot));
            bot = ++top;
        }
        return ret;
    }

    inline std::vector<std::string> splitSStringClean(const std::string &str, char delim)
    {
        std::vector<std::string> ret0 = splitSString(str, delim);
        std::vector<std::string> ret;
        for (auto &v : ret0)
            if (!v.empty())
                ret.push_back(v);
        return ret;
    }

    inline bool sstringHasSuffix(const std::string &str, const std::string &suffix)
    {
        if (str.size() < suffix.size())
            return false;
        return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    /// @brief Hash functor for std::vector<T>, combining element hashes via XOR.
    template <typename T>
    struct vector_hash
    {
        std::size_t operator()(const std::vector<T> &v) const noexcept
        {
            std::size_t r = 0;
            if constexpr (Meta::has_std_hash<T>::value)
                for (auto &i : v)
                    r = r ^ std::hash<T>{}(i);
            else if constexpr (Meta::is_std_vector<T>::value)
                for (auto &i : v)
                    r = r ^ vector_hash<typename T::value_type>{}(i);
            else
            {
                for (size_t i = 0; i < v.size() * sizeof(T); i++)
                    r = r ^ std::hash<uint8_t>{}(((uint8_t *)(v.data()))[i]);
            }
            return r;
        }

        template <class TBegin, class TEnd>
        std::size_t operator()(TBegin &&begin, TEnd &&end) const noexcept
        {
            std::size_t r = 0;
            ptrdiff_t size = end - begin;
            if constexpr (Meta::has_std_hash<T>::value)
                for (ptrdiff_t i = 0; i < size; i++)
                    r = r ^ std::hash<T>{}(begin[i]);
            else if constexpr (Meta::is_std_vector<T>::value)
                for (ptrdiff_t i = 0; i < size; i++)
                    r = r ^ vector_hash<typename T::value_type>{}(begin[i]);
            else
            {
                auto *start = reinterpret_cast<uint8_t *>(&(*begin));
                for (size_t i = 0; i < size * sizeof(T); i++)
                    r = r ^ std::hash<uint8_t>{}(start[i]);
            }
            return r;
        }
    };

    /// @brief Hash functor for std::array<T, s>, combining element hashes via XOR.
    template <typename T, std::size_t s>
    struct array_hash
    {
        std::size_t operator()(const std::array<T, s> &v) const noexcept
        {
            std::size_t r = 0;
            for (auto i : v)
            {
                r = r ^ std::hash<decltype(i)>()(i);
            }
            return r;
        }
    };

}

namespace DNDS::TermColor
{
    /// @brief ANSI escape: bright red foreground.
    constexpr std::string_view Red = "\033[91m";
    /// @brief ANSI escape: bright green foreground.
    constexpr std::string_view Green = "\033[92m";
    /// @brief ANSI escape: bright yellow foreground.
    constexpr std::string_view Yellow = "\033[93m";
    /// @brief ANSI escape: bright blue foreground.
    constexpr std::string_view Blue = "\033[94m";
    /// @brief ANSI escape: bright magenta foreground.
    constexpr std::string_view Magenta = "\033[95m";
    /// @brief ANSI escape: bright cyan foreground.
    constexpr std::string_view Cyan = "\033[96m";
    /// @brief ANSI escape: bright white foreground.
    constexpr std::string_view White = "\033[97m";
    /// @brief ANSI escape: reset all attributes.
    constexpr std::string_view Reset = "\033[0m";
    /// @brief ANSI escape: bold.
    constexpr std::string_view Bold = "\033[1m";
    /// @brief ANSI escape: underline.
    constexpr std::string_view Underline = "\033[4m";
    /// @brief ANSI escape: blinking text.
    constexpr std::string_view Blink = "\033[5m";
    /// @brief ANSI escape: reverse (swap fg/bg).
    constexpr std::string_view Reverse = "\033[7m";
    /// @brief ANSI escape: hidden text.
    constexpr std::string_view Hidden = "\033[8m";

}
/*









*/

namespace DNDS
{
    /// @brief Trivially-copyable empty placeholder type that accepts any assignment.
    struct Empty
    {
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(Empty, Empty)
        template <class T>
        DNDS_DEVICE_CALLABLE Empty &operator=(T v) { return *this; };
        template <class T>
        DNDS_DEVICE_CALLABLE Empty(T v) {}
    };

    /// @brief Empty placeholder type without a default constructor; accepts any assignment.
    struct EmptyNoDefault
    {
        DNDS_DEVICE_CALLABLE EmptyNoDefault(EmptyNoDefault &&v) = default;
        DNDS_DEVICE_CALLABLE EmptyNoDefault(const EmptyNoDefault &v) = default;
        DNDS_DEVICE_CALLABLE EmptyNoDefault &operator=(const EmptyNoDefault &) = default;
        template <class T>
        DNDS_DEVICE_CALLABLE EmptyNoDefault &operator=(T v) { return *this; };
        template <class T>
        DNDS_DEVICE_CALLABLE EmptyNoDefault(T v) {}
    };
}
/*









*/

namespace DNDS
{
    /// @brief Read/set the build version string accessible from code.
    /// @details When `ver` is non-empty it replaces the stored value; always
    /// returns the current value. Written into HDF5 attributes for traceability.
    std::string GetSetVersionName(const std::string &ver = "");
}

/*











*/

#if defined(_MSC_VER)
/// @brief `restrict`-pointer attribute (MSVC spelling).
#    define DNDS_RESTRICT __restrict
/// @brief Force-inline attribute (MSVC spelling).
#    define DNDS_FORCEINLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
/// @brief `restrict`-pointer attribute (GCC/Clang spelling).
#    define DNDS_RESTRICT __restrict__
/// @brief Force-inline attribute (GCC/Clang spelling).
#    define DNDS_FORCEINLINE inline __attribute__((always_inline))
#else
#    define DNDS_RESTRICT
#    define DNDS_FORCEINLINE
#endif
