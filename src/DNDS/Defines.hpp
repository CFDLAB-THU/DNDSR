#pragma once

// #define NDEBUG
#include "EigenPCH.hpp"
#include <cassert>
#include <cstdint>
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

#include <fmt/core.h>

#ifdef DNDS_USE_OMP
#include <omp.h>
#endif

#if defined(_WIN32) || defined(__WINDOWS_)
// #include <Windows.h>
// #include <process.h>
#endif

#include "Macros.hpp"
#include "Experimentals.hpp"
#include "Warnings.hpp"

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
#define DNDS_CURRENT_COMMIT_HASH UNKNOWN
#endif

#define DNDS_MACRO_TO_STRING(V) __DNDS_str(V)
#define __DNDS_str(V) #V

#ifdef __DNDS_REALLY_COMPILING__
#define DNDS_SWITCH_INTELLISENSE(real, intellisense) real
#else
#define DNDS_SWITCH_INTELLISENSE(real, intellisense) intellisense
#endif

#define DNDS_FMT_ARG(V) fmt::arg(#V, V)

/***************/ // DNDS_assertS

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

#ifdef DNDS_NDEBUG
#define DNDS_assert(expr)
#define DNDS_assert_info(expr, info)
#else
#define DNDS_assert(expr)    \
    (static_cast<bool>(expr) \
         ? void(0)           \
         : __DNDS_assert_false(#expr, __FILE__, __LINE__))
#define DNDS_assert_info(expr, info) \
    (static_cast<bool>(expr)         \
         ? void(0)                   \
         : __DNDS_assert_false_info(#expr, __FILE__, __LINE__, info))
#endif

extern "C" void DNDS_signal_handler(int signal);

namespace DNDS
{
    inline void RegisterSignalHandler()
    {
        std::signal(SIGSEGV, DNDS_signal_handler);
        std::signal(SIGABRT, DNDS_signal_handler);
        // std::signal(SIGKILL, DNDS_signal_handler);
    }
}

/***************/

static_assert(sizeof(uint8_t) == 1, "bad uint8_t");

namespace DNDS
{
    typedef double real;
    typedef int64_t index;
    typedef int32_t rowsize;
    typedef int64_t real_sized_index;
    typedef int32_t real_half_sized_index;
    static_assert(sizeof(real_sized_index) == sizeof(real) && sizeof(real_half_sized_index) == sizeof(real) / 2);

#define DNDS_INDEX_MAX INT64_MAX
#define DNDS_INDEX_MIN INT64_MIN
#define DNDS_ROWSIZE_MAX INT32_MAX
#define DNDS_ROWSIZE_MIN INT32_MIN

    static const char *outputDelim = "\t";

    template <typename T>
    using ssp = std::shared_ptr<T>;

    typedef std::vector<rowsize> t_RowsizeVec;
    typedef std::vector<index> t_IndexVec;
    typedef ssp<t_IndexVec> t_pIndexVec;

    typedef std::tuple<index, index> t_indexerPair;

    const index indexMin = INT64_MIN;

    const real UnInitReal = std::acos(-1) * 1e299 * std::sqrt(-1.0);
    const index UnInitIndex = INT64_MIN;
    static_assert(UnInitIndex < 0);
    const rowsize UnInitRowsize = INT32_MIN;
    static_assert(UnInitRowsize < 0);

    inline bool IsUnInitReal(real v)
    {
        // return (*(int64_t *)(&v)) == (*(int64_t *)(&UnInitReal));
        return std::isnan(v);
    }

    const real veryLargeReal = 3e200;
    const real largeReal = 3e10;
    const real verySmallReal = 1e-200;
    const real smallReal = 1e-10;

    const real pi = std::acos(-1);

    typedef Eigen::Matrix<real, -1, -1, Eigen::RowMajor> tDiFj;

    using MatrixXR = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXR = Eigen::Vector<real, Eigen::Dynamic>;
    using RowVectorXR = Eigen::RowVector<real, Eigen::Dynamic>;

/// TODO: change to template:
#define DNDS_MAKE_SSP(ssp, ...) (ssp = std::make_shared<typename decltype(ssp)::element_type>(__VA_ARGS__))

} // namespace DNDS

namespace DNDS
{
    const rowsize DynamicSize = -1;
    const rowsize NonUniformSize = -2;
    static_assert(DynamicSize != NonUniformSize, "DynamicSize, NonUniformSize definition conflict");

    inline constexpr int RowSize_To_EigenSize(rowsize rs)
    {
        return rs >= 0 ? static_cast<int>(rs) : (rs == DynamicSize || rs == NonUniformSize ? Eigen::Dynamic : INT_MIN);
    }
}

namespace DNDS
{
    extern std::ostream *logStream;

    extern bool useCout;

    std::ostream &log();

    bool logIsTTY();

    void setLogStream(std::ostream *nstream);
}


namespace DNDS
{
    extern int get_env_OMP_NUM_THREADS();
    extern int get_env_DNDS_DIST_OMP_NUM_THREADS();
}

/*









*/

namespace DNDS
{
    // Note that TtIndexVec being accumulated could overflow
    template <class TtRowsizeVec, class TtIndexVec>
    inline void AccumulateRowSize(const TtRowsizeVec &rowsizes, TtIndexVec &rowstarts)
    {
        static_assert(std::is_signed_v<typename TtIndexVec::value_type>, "row starts should be signed");
        rowstarts.resize(rowsizes.size() + 1);
        rowstarts[0] = 0;
        for (typename TtIndexVec::size_type i = 1; i < rowstarts.size(); i++)
        {
            rowstarts[i] = rowstarts[i - 1] + rowsizes[i - 1];
            DNDS_assert(rowstarts[i] >= 0);
        }
    }

    template <class T>
    inline bool checkUniformVector(const std::vector<T> &dat, T &value)
    {
        if (dat.size() == 0)
            return false;
        value = dat[0];
        for (auto i = 1; i < dat.size(); i++)
            if (dat[i] != value)
                return false;
        return true;
    }

    template <class T, class TP = T>
    inline void PrintVec(const std::vector<T> &dat, std::ostream &out)
    {
        for (auto i = 0; i < dat.size(); i++)
            out << TP(dat[i]) << outputDelim;
    }

    /// \brief l must be non-negative, r must be positive. integers
    template <class TL, class TR>
    inline constexpr auto divCeil(TL l, TR r)
    {
        return l / r + (l % r) ? 1 : 0;
    }
}

/*









*/

namespace DNDS
{
    template <typename T>
    inline constexpr T sqr(const T &a)
    {
        static_assert(std::is_arithmetic_v<T>, "need arithmetic");
        return a * a;
    }

    template <typename T>
    inline constexpr T cube(const T &a)
    {
        static_assert(std::is_arithmetic_v<T>, "need arithmetic");
        return a * a * a;
    }

    inline constexpr real sign(real a)
    {
        return a > 0 ? 1 : (a < 0 ? -1 : 0);
    }

    inline constexpr real signTol(real a, real tol)
    {
        return a > tol ? 1 : (a < -tol ? -1 : 0);
    }

    inline constexpr real signP(real a)
    {
        return a >= 0 ? 1 : -1;
    }

    inline constexpr real signM(real a)
    {
        return a <= 0 ? -1 : 1;
    }

    template <typename T>
    inline constexpr T mod(T a, T b)
    {
        static_assert(std::is_signed<T>::value && std::is_integral<T>::value, "not legal mod type");
        T val = a % b;
        if (val < 0)
            val += b;
        return val;
    }

    template <typename T>
    inline constexpr T divide_ceil(T a, T b)
    {
        static_assert(std::is_integral<T>::value, "not legal mod type");
        return a / b + (a % b ? 1 : 0);
    }
    // const int a = divide_ceil(23, 11);

    /**
     * \param b should be positive
     */
    inline real float_mod(real a, real b)
    {
        return a - std::floor(a / b) * b;
    }

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

    inline int32_t checkedIndexTo32(index v)
    {
        DNDS_assert_info(!(v > static_cast<index>(INT32_MAX) || v < static_cast<index>(INT32_MIN)),
                         fmt::format("Index {} to int32 overflow", v));
        return static_cast<int32_t>(v);
    }

    std::string getStringForceWString(const std::wstring &v);

    inline std::string getStringForcePath(const std::filesystem::path::string_type &v)
    {
#ifdef _WIN32
        return getStringForceWString(v);
#else
        return std::string{v};
#endif
    }

}

/*-----------------------------------------*/
// some meta-programming utilities
namespace DNDS
{
    namespace Meta
    {
        template <class T>
        struct is_std_array : std::false_type
        {
        };

        template <class T, size_t N>
        struct is_std_array<std::array<T, N>> : std::true_type
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
    }
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
            if (v.size())
                ret.push_back(v);
        return ret;
    }
}
template <typename T>
struct std::hash<std::vector<T>>
{
    std::size_t operator()(const std::vector<T> &v) const noexcept
    {
        std::size_t r = 0;
        for (auto i : v)
        {
            r = r ^ std::hash<decltype(i)>()(i);
        }
        return r;
    }
};

template <typename T, std::size_t s>
struct std::hash<std::array<T, s>>
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

namespace DNDS::TermColor
{
    constexpr std::string_view Red = "\033[91m";
    constexpr std::string_view Green = "\033[92m";
    constexpr std::string_view Yellow = "\033[93m";
    constexpr std::string_view Blue = "\033[94m";
    constexpr std::string_view Magenta = "\033[95m";
    constexpr std::string_view Cyan = "\033[96m";
    constexpr std::string_view White = "\033[97m";
    constexpr std::string_view Reset = "\033[0m";
    constexpr std::string_view Bold = "\033[1m";
    constexpr std::string_view Underline = "\033[4m";
    constexpr std::string_view Blink = "\033[5m";
    constexpr std::string_view Reverse = "\033[7m";
    constexpr std::string_view Hidden = "\033[8m";

}

/*









*/
