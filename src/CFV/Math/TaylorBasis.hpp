#pragma once

#include "CFV/Math/PolynomialReconstruction.hpp"
#include "DNDS/Math/ComputeScalar.hpp"

namespace DNDS::CFV::Math
{
    /// Point-local arithmetic shared by all Taylor modes, derivatives and field
    /// components. Stores only x, x^2, x^3 in each normalized coordinate and
    /// scale^{-1..-3}: twelve scalars, with no allocation or geometry owner.
    /// Prepare once per physical point after validating a positive cell scale.
    template <class Scalar>
    struct TaylorPointEvaluationT
    {
        std::array<std::array<Scalar, 3>, 3> coordinatePowers{};
        std::array<Scalar, 3> inverseScalePowers{};

        DNDS_DEVICE_CALLABLE static TaylorPointEvaluationT FromNormalized(
            const std::array<Scalar, 3> &x, Scalar scale) noexcept
        {
            return FromNormalizedWithInverseScale(x, Scalar(1) / scale);
        }
        /// Internal common constructor; callers that already normalized the
        /// point supply the same inverse scale without a second division.
        DNDS_DEVICE_CALLABLE static TaylorPointEvaluationT FromNormalizedWithInverseScale(
            const std::array<Scalar, 3> &x, Scalar inverseScale) noexcept
        {
            TaylorPointEvaluationT result;
            for (int d = 0; d < 3; ++d)
            {
                result.coordinatePowers[d][0] = x[d];
                result.coordinatePowers[d][1] = x[d] * x[d];
                result.coordinatePowers[d][2] = result.coordinatePowers[d][1] * x[d];
            }
            result.inverseScalePowers = {inverseScale, inverseScale * inverseScale,
                                         inverseScale * inverseScale * inverseScale};
            return result;
        }
        template <class InputScalar>
        DNDS_DEVICE_CALLABLE static TaylorPointEvaluationT FromPoint(
            const std::array<InputScalar, 3> &point, const std::array<InputScalar, 3> &center, InputScalar scale) noexcept
        {
            std::array<Scalar, 3> x{};
            const Scalar inverseScale = Scalar(1) / DNDS::Math::ScalarCast<Scalar>(scale);
            for (int d = 0; d < 3; ++d)
                x[d] = (DNDS::Math::ScalarCast<Scalar>(point[d]) - DNDS::Math::ScalarCast<Scalar>(center[d])) * inverseScale;
            return FromNormalizedWithInverseScale(x, inverseScale);
        }
        DNDS_DEVICE_CALLABLE Scalar CoordinatePower(int direction, int exponent) const noexcept
        {
            return exponent == 0 ? Scalar(1) : coordinatePowers[direction][exponent - 1];
        }
        DNDS_DEVICE_CALLABLE Scalar DerivativeScale(int order) const noexcept
        {
            return order == 0 ? Scalar(1) : inverseScalePowers[order - 1];
        }
    };
    /// FP64 canonical spelling, preserving existing CPU/CUDA call sites.
    using TaylorPointEvaluation = TaylorPointEvaluationT<real>;
    /// Value-only polynomial ordering for device evaluation. Prepared from the
    /// canonical Geom tables, rather than duplicating their 2D/3D mode order.
    /// Coordinates are dimensionless (point-center)/scale; physical derivative
    /// factors use the same isotropic scale. No factorial normalization of the
    /// monomials is introduced. The zero-mean correction belongs to the caller.
    struct TaylorBasisParameters
    {
        std::array<std::array<int, 3>, 20> orders{};
        int dimension = 0, degree = 0, size = 0;

        DNDS_DEVICE_CALLABLE static real Power(real x, int exponent)
        {
            return exponent == 0 ? 1 : exponent == 1 ? x
                                   : exponent == 2   ? x * x
                                                     : x * x * x;
        }
        /// Evaluate a physical derivative with point-local powers already
        /// prepared. Canonical mode and derivative ordinals include constants.
        template <class Scalar>
        DNDS_DEVICE_CALLABLE Scalar Derivative(int derivative, int mode, const TaylorPointEvaluationT<Scalar> &point) const noexcept
        {
            const auto &alpha = orders[mode], &beta = orders[derivative];
            int factor = 1;
            for (int d = 0; d < 3; ++d)
            {
                if (beta[d] > alpha[d])
                    return Scalar(0);
                for (int k = 0; k < beta[d]; ++k)
                    factor *= alpha[d] - k;
            }
            const Scalar numerator = Scalar(factor) * point.CoordinatePower(0, alpha[0] - beta[0]) *
                                     point.CoordinatePower(1, alpha[1] - beta[1]) * point.CoordinatePower(2, alpha[2] - beta[2]);
            return numerator * point.DerivativeScale(beta[0] + beta[1] + beta[2]);
        }
        /// Scalar convenience API. Hot loops should prepare once and use the
        /// context overload instead of rebuilding powers for every mode.
        DNDS_DEVICE_CALLABLE real Derivative(int derivative, int mode, const std::array<real, 3> &x, real scale) const noexcept
        {
            return Derivative(derivative, mode, TaylorPointEvaluation::FromNormalized(x, scale));
        }
    };
    /// Central definition of polynomial/derivative ordering and Cartesian frame
    /// changes for the scalar-scaled Taylor family. Geometry supplies centers,
    /// lengths and moments; a basis owns none of them. Cartesian-tree backends
    /// can specialize reference evaluation while preserving this mode contract.
    class TaylorBasis
    {
        int _dimension, _degree;

    public:
        TaylorBasis(int dimension, int degree) : _dimension(dimension), _degree(degree)
        {
            (void)PolynomialSize(dimension, degree);
        }
        int Size() const { return PolynomialSize(_dimension, _degree); }
        TaylorBasisParameters KernelParameters() const
        {
            TaylorBasisParameters result;
            result.dimension = _dimension;
            result.degree = _degree;
            result.size = Size();
            for (int mode = 0; mode < result.size; ++mode)
                result.orders[mode] = MultiIndex(mode);
            return result;
        }
        const std::array<int, 3> &MultiIndex(int mode) const
        {
            DNDS_check_throw_info(mode >= 0 && mode < Size(), "Taylor derivative mode outside basis");
            return _dimension == 2 ? Geom::Base::diffOperatorOrderList2D[mode] : Geom::Base::diffOperatorOrderList[mode];
        }
        int Degree(int mode) const
        {
            const auto &alpha = MultiIndex(mode);
            return alpha[0] + alpha[1] + alpha[2];
        }
        void Derivatives(SmallMatrix &values, const Geom::tPoint &point, const Geom::tPoint &center, real scale) const
        {
            PolynomialDerivatives(values, _dimension, _degree, point, center, scale);
        }
        int DerivativeNumber(int order) const
        {
            DNDS_check_throw_info(order >= 0 && order <= 3, "invalid cached derivative order");
            return order == 0 ? 1 : PolynomialSize(_dimension, std::min(order, _degree));
        }
        /// Shared zero-mean reconstruction basis for face and arbitrary-point
        /// sampling. Scratch is bounded; no persistent cache or allocation here.
        template <class MomentRow>
        void Reconstruction(SmallMatrix &values, const Geom::tPoint &point,
                            const Geom::tPoint &center, real scale,
                            const MomentRow &moments, int order) const
        {
            SmallMatrix full;
            (void)DerivativeNumber(order);
            PolynomialDerivatives(full, _dimension, _degree, point, center, scale, order);
            values = full.rightCols(Size() - 1);
            for (int mode = 0; mode < Size() - 1; ++mode)
                values(0, mode) -= moments[mode + 1];
        }
        /// Pull derivatives of f(y), y=J*x+b, into the x frame. Each row is
        /// one unique symmetric derivative; expansion supplies multiplicities
        /// in the chain rule. All derivative orders are transformed together.
        /// Translation is already handled by evaluating at the mapped point.
        void Pullback(SmallMatrix &values, const Geom::tGPoint &jacobian) const
        {
            DNDS_check_throw_info(values.rows() == Size() && jacobian.allFinite(), "invalid Taylor derivative pullback");
            if (_dimension == 2)
                DNDS_check_throw_info((jacobian.block<1, 2>(2, 0).norm() < 1e-12 &&
                                       jacobian.block<2, 1>(0, 2).norm() < 1e-12),
                                      "2D basis rotation leaves its geometric plane");
            SmallMatrix transform;
            transform.setZero(Size(), Size());
            for (int row = 0; row < Size(); ++row)
            {
                std::array<real, 20> coefficients{}, next{};
                coefficients[0] = 1;
                const auto &alpha = MultiIndex(row);
                for (int direction = 0; direction < _dimension; ++direction)
                    for (int repeat = 0; repeat < alpha[direction]; ++repeat)
                    {
                        next.fill(0);
                        for (int mode = 0; mode < Size(); ++mode)
                            if (coefficients[mode] != 0)
                                for (int target = 0; target < _dimension; ++target)
                                {
                                    auto beta = MultiIndex(mode);
                                    ++beta[target];
                                    for (int result = 0; result < Size(); ++result)
                                        if (MultiIndex(result) == beta)
                                        {
                                            next[result] += coefficients[mode] * jacobian(target, direction);
                                            break;
                                        }
                                }
                        coefficients = next;
                    }
                for (int col = 0; col < Size(); ++col)
                    transform(row, col) = coefficients[col];
            }
            values = (transform * values).eval();
        }
    };

    /// Borrowed local matrix, including subsets of a larger derivative cache.
    /// Retain its read scope or caller-owned scratch through use.
    struct ReconstructionBasisView
    {
        const real *values = nullptr;
        int derivatives = 0, modes = 0, columnStride = 0;
        DNDS_DEVICE_CALLABLE real operator()(int derivative, int mode) const
        {
            return values[mode * columnStride + derivative];
        }
    };
}
