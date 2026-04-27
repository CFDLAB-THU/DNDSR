#pragma once
/// @file EigenUtil.hpp
/// @brief Eigen extensions: `to_string`, an fmt-safe wrapper, and fmt formatter
/// specialisations for dense Eigen matrices.
///
/// The @ref DNDS::MatrixFMTSafe "MatrixFMTSafe" / @ref VectorFMTSafe helpers exist because modern fmtlib
/// (with `fmt/ranges.h`) will detect `Eigen::Matrix` as a range and format
/// it as `[a, b, c, ...]`, overriding the Eigen stream formatting that
/// DNDSR wants. Wrapping Eigen types with these aliases hides the iterator
/// interface from fmt and keeps the matrix pretty-print.

#include "EigenPCH.hpp"

#include "Defines.hpp"

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/format.h>
#include "Device/DeviceStorage.hpp"
#include "Vector.hpp"

namespace DNDS
{
    /// @brief Render an `Eigen::DenseBase` to a string via `operator<<`.
    /// @param precision Setprecision applied if `> 0`.
    /// @param scientific Whether to use `std::scientific` notation.
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
    /**
     * @brief `Eigen::Matrix` wrapper that hides `begin`/`end` from fmt.
     *
     * @details When both `fmt/ranges.h` and Eigen are present, the fmt range
     * formatter picks up `Eigen::Matrix` via its iterator interface and renders
     * it as a bracketed list (`[1, 2, 3]`). This disables Eigen's stream
     * formatting path. This wrapper inherits from `Eigen::Matrix` but deletes
     * `begin` / `end`, so fmt falls back to the ostream formatter.
     *
     * Use this type (or its @ref VectorFMTSafe / @ref RowVectorFMTSafe aliases) wherever
     * Eigen objects need to pass through `fmt::format`.
     */
    template <class T, int M, int N, int options = AutoAlign | ((M == 1 && N != 1) ? Eigen ::RowMajor : !(M == 1 && N != 1) ? Eigen ::ColMajor
                                                                                                                            : Eigen ::ColMajor),
              int max_m = M, int max_n = N>
    struct MatrixFMTSafe : public Matrix<T, M, N, options, max_m, max_n>
    {
        using Base = Matrix<T, M, N, options, max_m, max_n>;
        using Base::Base;

        /// @brief Deleted to hide range interface from fmt.
        void begin() = delete;
        /// @brief Deleted to hide range interface from fmt.
        void end() = delete;
    };

    /// @brief Column-vector alias of @ref DNDS::MatrixFMTSafe "MatrixFMTSafe".
    template <class T, int M>
    using VectorFMTSafe = MatrixFMTSafe<T, M, 1>;

    /// @brief Row-vector alias of @ref DNDS::MatrixFMTSafe "MatrixFMTSafe".
    template <class T, int N>
    using RowVectorFMTSafe = MatrixFMTSafe<T, 1, N>;
}

namespace DNDS::Meta
{
    /// @brief Type trait: is `T` a @ref DNDS::MatrixFMTSafe "MatrixFMTSafe" with real scalar? Used by
    /// the fmt formatter below to catch both wrapped and unwrapped Eigen types.
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

namespace DNDS
{

    template <DeviceBackend B, typename t_scalar, int M, int N>
    class EigenMatrixView
    {
    public:
        static_assert(M >= 0 || M == Eigen::Dynamic, "M needs to be a valid eigen size");
        static_assert(N >= 0 || N == Eigen::Dynamic, "N needs to be a valid eigen size");
        using t_matrix = Eigen::Matrix<std::remove_cv_t<t_scalar>, M, N>;
        using t_map_const = Eigen::Map<const t_matrix>;
        using t_map = std::conditional_t<std::is_const_v<t_scalar>,
                                         t_map_const, Eigen::Map<t_matrix>>;
        using t_self = EigenMatrixView<B, t_scalar, M, N>;

    protected:
        t_scalar *ptr = nullptr;
        std::conditional_t<M >= 0, EmptyNoDefault, rowsize> M_dynamic = 0;
        std::conditional_t<N >= 0, EmptyNoDefault, rowsize> N_dynamic = 0;

    public:
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(EigenMatrixView, t_self)

        DNDS_DEVICE_CALLABLE EigenMatrixView(t_scalar *n_ptr, rowsize m, rowsize n)
            : ptr(n_ptr), M_dynamic(m), N_dynamic(n)
        {
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] t_scalar *data() const
        {
            return ptr;
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] rowsize rows() const
        {
            if constexpr (M >= 0)
                return M;
            else
                return M_dynamic;
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] rowsize cols() const
        {
            if constexpr (N >= 0)
                return N;
            else
                return N_dynamic;
        }

        DNDS_DEVICE_CALLABLE t_map map()
        {
            return {ptr, rows(), cols()};
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] t_map_const map() const
        {
            return {ptr, rows(), cols()};
        }
    };

    static_assert(std::is_trivially_copyable_v<EigenMatrixView<DeviceBackend::Host, real, Eigen::Dynamic, Eigen::Dynamic>>);

    template <typename t_scalar, int M, int N>
    class HostDeviceEigenMatrix
    {
    public:
        using t_matrix = Eigen::Matrix<t_scalar, M, N>;
        using t_map = Eigen::Map<t_matrix>;

    protected:
        rowsize M_dynamic = 0;
        rowsize N_dynamic = 0;
        host_device_vector<t_scalar> h_data;

    public:
        HostDeviceEigenMatrix()
        {
            h_data.resize(this->size());
        }

        [[nodiscard]] rowsize rows() const
        {
            if constexpr (M >= 0)
                return M;
            else
                return M_dynamic;
        }

        [[nodiscard]] rowsize cols() const
        {
            if constexpr (N >= 0)
                return N;
            else
                return N_dynamic;
        }

        [[nodiscard]] rowsize size() const
        {
            return rows() * cols();
        }

        void resize(int m, int n)
        {
            if constexpr (M >= 0)
                DNDS_assert(M == m);
            else
                M_dynamic = m;
            if constexpr (N >= 0)
                DNDS_assert(N == n);
            else
                N_dynamic = n;
            h_data.resize(this->size());
        }

        t_map map()
        {
            return {h_data.data(), this->rows(), this->cols()};
        }

        template <DeviceBackend B>
        using t_deviceView = EigenMatrixView<B, t_scalar, M, N>;

        void to_host()
        {
            h_data.to_host();
        }

        void to_device(DeviceBackend B)
        {
            h_data.to_device(B);
        }

        template <DeviceBackend B>
        t_deviceView<B> deviceView()
        {
            DNDS_assert_info(h_data.device() == B || (h_data.device() == DeviceBackend::Unknown && B == DeviceBackend::Host),
                             "data not on this backend");
            switch (B)
            {
            case DeviceBackend::Unknown:
            case DeviceBackend::Host:
                return t_deviceView<B>{h_data.data(), rows(), cols()};
#ifdef DNDS_USE_CUDA
            case DeviceBackend::CUDA:
                return t_deviceView<B>{h_data.dataDevice(), rows(), cols()};
#endif
            default:
                DNDS_assert_info(false, std::string("this device not implemented -- ") + device_backend_name(B));
                return t_deviceView<B>{h_data.dataDevice(), rows(), cols()};
            }
        }
    };
}
