#pragma once

#include "Defines.hpp"

#include <fmt/core.h>
#include <Eigen/Core>
#include <fmt/ostream.h>
#include <fmt/format.h>

namespace DNDS
{
    // TODO: lessen copying chance?
    template <class dir>
    std::string to_string(const Eigen::DenseBase<dir> &v,
                          int precision = 5,
                          bool scientific = false)
    {
        std::stringstream ss;
        if (precision > 0)
            ss << std::setprecision(precision);
        if (scientific)
            ss << std::scientific;
        ss << v;
        return ss.str();
    }

}

namespace Eigen
{
    template <class T, int M, int N, int options = AutoAlign | ((M == 1 && N != 1) ? Eigen ::RowMajor : !(M == 1 && N != 1) ? Eigen ::ColMajor
                                                                                                                            : Eigen ::ColMajor),
              int max_m = M, int max_n = N>
    struct MatrixFMTSafe : public Matrix<T, M, N, options, max_m, max_n>
    {
        using Base = Matrix<T, M, N, options, max_m, max_n>;
        using Base::Base;

        void begin() = delete;
        void end() = delete;
    };

    template <class T, int M>
    using VectorFMTSafe = MatrixFMTSafe<T, M, 1>;

    template <class T, int N>
    using RowVectorFMTSafe = MatrixFMTSafe<T, 1, N>;
}

namespace DNDS::Meta
{
    template <class T>
    struct is_real_eigen_fmt_safe_matrix : public std::false_type
    {
    };

    template <int M, int N, int options, int max_m, int max_n>
    struct is_real_eigen_fmt_safe_matrix<Eigen::MatrixFMTSafe<real, M, N, options, max_m, max_n>> : public std::true_type
    {
    };

    template <class T>
    constexpr bool is_real_eigen_fmt_safe_matrix_v = is_real_eigen_fmt_safe_matrix<T>::value;

    const bool v = is_real_eigen_fmt_safe_matrix_v<Eigen::MatrixFMTSafe<real, 3, 3>>;
}

// formatter support for dense eigen matrices
// ! is not compatible with fmt/ranges.h
// ! use Eigen::MatrixFMTSafe if fmt/ranges.h is present
// ! Eigen::Vector s would be fine if fmt/ranges.h is present, but using fmt/ranges.h syntax
template <typename T, typename Char>
struct fmt::formatter<T, Char,
                      std::enable_if_t<DNDS::Meta::is_eigen_dense_v<std::remove_cv_t<T>> ||
                                       DNDS::Meta::is_real_eigen_fmt_safe_matrix_v<std::remove_cv_t<T>>>>
// template <int M, int N, int options, int max_m, int max_n, class Char>
// struct fmt::formatter<Eigen::Matrix<DNDS::real, M, N, options, max_m, max_n>, Char>
{
    // using TMat = Eigen::Matrix<DNDS::real, M, N, options, max_m, max_n>;
    using TMat = std::remove_cv_t<T>;
    char align = '>';
    char sign = ' ';
    int width = -1;
    int precision = 16;
    char type = 'g';
    std::string formatSpecC = "{}";

    auto parse(fmt::format_parse_context &ctx)
    {
        auto it = ctx.begin(), end = ctx.end();
        bool afterDot = false;
        while (it != end && *it != '}')
        { // a home-cooked version of float point format parser
            switch (*it)
            {
            case '<':
            case '>':
            case '^':
                align = *it++;
                break;
            case '+':
            case '-':
            case ' ':
                sign = *it++;
                break;
            case 'e':
            case 'E':
            case 'f':
            case 'F':
            case 'g':
            case 'G':
                type = *it++;
                break;
            case '.':
                afterDot = true;
                it++;
                break;
            default:
            {
                if (*it >= '0' && *it <= '9')
                {
                    std::string v;
                    v.reserve(20);
                    while (it != end && *it >= '0' && *it <= '9')
                        v.push_back(*it++);
                    if (afterDot)
                        precision = std::stoi(v);
                    else
                        width = std::stoi(v);
                }
                else
                    DNDS_assert_info(false, fmt::format("invalid char {}", *it));
            }
            break;
            }
        }
        if (width == -1)
            formatSpecC = fmt::format(FMT_STRING("{{:{0}{1}.{3}{4}}}"), align, sign, width, precision, type);
        else
            formatSpecC = fmt::format(FMT_STRING("{{:{0}{1}{2}.{3}{4}}}"), align, sign, width, precision, type);
        return it;
    }

    auto format(const TMat &mat, fmt::format_context &ctx) const
    {
        std::string buf;
        buf.reserve(mat.size() * 10);
        buf.push_back('[');
        for (Eigen::Index i = 0; i < mat.rows(); ++i)
        {
            for (Eigen::Index j = 0; j < mat.cols(); ++j)
            {
                if (j > 0)
                    buf.append(",");
                fmt::format_to(std::back_inserter(buf), formatSpecC, mat(i, j));
            }
            if (i < mat.rows() - 1)
                buf.append(";\n");
        }
        buf.push_back(']');
        return fmt::format_to(ctx.out(), "{}", buf);
    }
};
