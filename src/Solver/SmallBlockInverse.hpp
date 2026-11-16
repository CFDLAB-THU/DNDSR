#pragma once

#include "Solver/InverseRefinement.hpp"
#include <cmath>
#include <limits>

namespace DNDS::Direct
{
    /// Euler/Cauchy's normalized-block LU acceptance threshold. This is a
    /// reciprocal 1-norm condition estimate, not a pivot-size or determinant test.
    inline constexpr real DiagonalLUConditionMinimum = 1e-7;

    template <int N>
    struct SmallDenseMatrix
    {
        real values[N * N];
        DNDS_DEVICE_CALLABLE real &operator()(int i, int j) { return values[i + N * j]; }
        DNDS_DEVICE_CALLABLE real operator()(int i, int j) const { return values[i + N * j]; }
    };

    /** @brief Allocation-free partial-pivot LU for the 5/6/7 block experiment.
     * @details P*A=L*U. Solve and transpose-solve use the same factorization.
     * ConditionReciprocal implements the Hager/Higham 1-norm estimator used by
     * Eigen PartialPivLU::rcond: at most four sign/simplex iterations followed
     * by the alternating-vector safeguard. Thus the CPU guard is preserved;
     * using pivot ratios or an exact inverse norm here would change the policy.
     */
    template <int N>
    struct SmallPartialPivLU
    {
        SmallDenseMatrix<N> factors;
        int permutation[N];
        real normOne = 0;

        DNDS_DEVICE_CALLABLE bool Factor(const SmallDenseMatrix<N> &a)
        {
            normOne = 0;
            for (int j = 0; j < N; ++j)
            {
                real sum = 0;
                permutation[j] = j;
                for (int i = 0; i < N; ++i)
                {
                    factors(i, j) = a(i, j);
                    sum += std::abs(a(i, j));
                }
                normOne = normOne > sum ? normOne : sum;
            }
            for (int k = 0; k < N; ++k)
            {
                int pivot = k;
                for (int i = k + 1; i < N; ++i)
                    if (std::abs(factors(i, k)) > std::abs(factors(pivot, k)))
                        pivot = i;
                if (!(std::isfinite(factors(pivot, k)) && factors(pivot, k) != 0))
                    return false;
                if (pivot != k)
                {
                    const int old = permutation[k];
                    permutation[k] = permutation[pivot];
                    permutation[pivot] = old;
                    for (int j = 0; j < N; ++j)
                    {
                        const real oldValue = factors(k, j);
                        factors(k, j) = factors(pivot, j);
                        factors(pivot, j) = oldValue;
                    }
                }
                for (int i = k + 1; i < N; ++i)
                {
                    factors(i, k) /= factors(k, k);
                    for (int j = k + 1; j < N; ++j)
                        factors(i, j) -= factors(i, k) * factors(k, j);
                }
            }
            return true;
        }

        DNDS_DEVICE_CALLABLE void Solve(real (&v)[N], bool transpose = false) const
        {
            real temporary[N];
            if (!transpose)
            {
                for (int i = 0; i < N; ++i)
                    temporary[i] = v[permutation[i]];
                for (int i = 0; i < N; ++i)
                {
                    v[i] = temporary[i];
                    for (int j = 0; j < i; ++j)
                        v[i] -= factors(i, j) * v[j];
                }
                for (int i = N - 1; i >= 0; --i)
                {
                    for (int j = i + 1; j < N; ++j)
                        v[i] -= factors(i, j) * v[j];
                    v[i] /= factors(i, i);
                }
            }
            else
            {
                for (int i = 0; i < N; ++i)
                {
                    for (int j = 0; j < i; ++j)
                        v[i] -= factors(j, i) * v[j];
                    v[i] /= factors(i, i);
                }
                for (int i = N - 1; i >= 0; --i)
                    for (int j = i + 1; j < N; ++j)
                        v[i] -= factors(j, i) * v[j];
                for (int i = 0; i < N; ++i)
                    temporary[permutation[i]] = v[i];
                for (int i = 0; i < N; ++i)
                    v[i] = temporary[i];
            }
        }
        DNDS_DEVICE_CALLABLE static real OneNorm(const real (&v)[N])
        {
            real value = 0;
            for (int i = 0; i < N; ++i)
                value += std::abs(v[i]);
            return value;
        }
        DNDS_DEVICE_CALLABLE real ConditionReciprocal() const
        {
            real v[N], signs[N], oldSigns[N]{};
            for (int i = 0; i < N; ++i)
                v[i] = real(1) / N;
            Solve(v);
            real lower = OneNorm(v), oldLower = lower;
            int oldIndex = -1;
            for (int iteration = 0; iteration < 4; ++iteration)
            {
                bool same = iteration != 0;
                for (int i = 0; i < N; ++i)
                {
                    signs[i] = v[i] < 0 ? -1 : 1;
                    same = same && signs[i] == oldSigns[i];
                    v[i] = signs[i];
                }
                if (same)
                    break;
                Solve(v, true);
                int next = 0;
                for (int i = 1; i < N; ++i)
                    if (std::abs(v[i]) > std::abs(v[next]))
                        next = i;
                if (next == oldIndex)
                    break;
                for (int i = 0; i < N; ++i)
                    v[i] = i == next ? 1 : 0;
                Solve(v);
                lower = OneNorm(v);
                if (lower <= oldLower)
                    break;
                for (int i = 0; i < N; ++i)
                    oldSigns[i] = signs[i];
                oldIndex = next;
                oldLower = lower;
            }
            for (int i = 0; i < N; ++i)
                v[i] = (i % 2 ? -1 : 1) * (1 + real(i) / (N - 1));
            Solve(v);
            const real alternate = 2 * OneNorm(v) / (3 * N);
            const real estimate = lower > alternate ? lower : alternate;
            return estimate == 0 ? 0 : (1 / estimate) / normOne;
        }
    };

    /// Invert an already row-normalized block. False requests SVD, never a
    /// regularization, identity substitute, or change to the condition guard.
    template <int N, bool Refine>
    DNDS_DEVICE_CALLABLE bool TrySmallLUInverse(const SmallDenseMatrix<N> &a,
                                                SmallDenseMatrix<N> &inverse, int steps = 0)
    {
        SmallPartialPivLU<N> lu;
        if (!lu.Factor(a))
            return false;
        const real rcond = lu.ConditionReciprocal();
        if (!(std::isfinite(rcond) && rcond >= DiagonalLUConditionMinimum))
            return false;
        for (int j = 0; j < N; ++j)
        {
            real v[N];
            for (int i = 0; i < N; ++i)
                v[i] = i == j ? 1 : 0;
            lu.Solve(v);
            for (int i = 0; i < N; ++i)
            {
                if (!std::isfinite(v[i]))
                    return false;
                inverse(i, j) = v[i];
            }
        }
        if constexpr (Refine)
        {
            SmallDenseMatrix<N> residual, trial;
            RefineInverse(a, inverse, residual, trial, N, steps);
        }
        return true;
    }

    /** @brief Filtered one-sided Jacobi SVD fallback for small normalized blocks.
     * @details Orthogonalize columns of B and accumulate right vectors V.
     * Singular values are column norms and U contains normalized columns.
     * Scaling by max(abs(B)) avoids overflow in dot products. Retain precisely
     * sigma > epsilon*sigma_max, including the same strict inequality as CPU.
     * No diagonal perturbation is allowed. Failure to converge in 100 sweeps
     * returns false; the wrapper publishes a numerical error, not an LU inverse.
     */
    template <int N, bool Refine>
    DNDS_DEVICE_CALLABLE bool SmallSVDInverse(const SmallDenseMatrix<N> &a,
                                              SmallDenseMatrix<N> &inverse, int &truncated, int steps = 0)
    {
        SmallDenseMatrix<N> b, v;
        real scale = 0;
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                scale = scale > std::abs(a(i, j)) ? scale : std::abs(a(i, j));
        if (!(std::isfinite(scale) && scale > 0))
            return false;
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
            {
                b(i, j) = a(i, j) / scale;
                v(i, j) = i == j ? 1 : 0;
            }
        bool converged = false;
        for (int sweep = 0; sweep < 100; ++sweep)
        {
            bool changed = false;
            for (int p = 0; p < N; ++p)
                for (int q = p + 1; q < N; ++q)
                {
                    real alpha = 0, beta = 0, gamma = 0;
                    for (int i = 0; i < N; ++i)
                    {
                        alpha += b(i, p) * b(i, p);
                        beta += b(i, q) * b(i, q);
                        gamma += b(i, p) * b(i, q);
                    }
                    // FMA can leave subnormal null-column remnants where the
                    // relative orthogonality threshold underflows to zero.
                    // Treat dot products below the normal-number floor as
                    // converged rather than rotating forever on that noise.
                    // This is an arithmetic convergence floor; the singular
                    // filter below remains exactly epsilon*sigma_max.
                    const real relative = std::numeric_limits<real>::epsilon() * std::sqrt(alpha) * std::sqrt(beta);
                    const real threshold = relative > std::numeric_limits<real>::min() ? relative : std::numeric_limits<real>::min();
                    if (std::abs(gamma) <= threshold)
                        continue;
                    const real delta = beta - alpha;
                    const real tau = delta / (2 * gamma);
                    const real t = std::isfinite(tau)
                                       ? std::copysign(real(1), tau) / (std::abs(tau) + std::hypot(real(1), tau))
                                       : gamma / delta;
                    const real c = 1 / std::sqrt(1 + t * t), s = t * c;
                    for (int i = 0; i < N; ++i)
                    {
                        const real bp = b(i, p), vp = v(i, p);
                        b(i, p) = c * bp - s * b(i, q);
                        b(i, q) = s * bp + c * b(i, q);
                        v(i, p) = c * vp - s * v(i, q);
                        v(i, q) = s * vp + c * v(i, q);
                    }
                    changed = true;
                }
            if (!changed)
            {
                converged = true;
                break;
            }
        }
        if (!converged)
            return false;
        real singular[N], maximum = 0;
        for (int k = 0; k < N; ++k)
        {
            real norm = 0;
            for (int i = 0; i < N; ++i)
                norm = std::hypot(norm, b(i, k));
            singular[k] = norm;
            maximum = maximum > norm ? maximum : norm;
        }
        const real cutoff = std::numeric_limits<real>::epsilon() * maximum;
        truncated = 0;
        for (int k = 0; k < N; ++k)
            truncated += singular[k] <= cutoff;
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
            {
                real value = 0;
                for (int k = 0; k < N; ++k)
                    if (singular[k] > cutoff)
                        value += (v(i, k) / singular[k]) * (b(j, k) / singular[k]);
                inverse(i, j) = value / scale;
                if (!std::isfinite(inverse(i, j)))
                    return false;
            }
        if constexpr (Refine)
        {
            SmallDenseMatrix<N> u, c, z, residual, trial;
            for (int k = 0; k < N; ++k)
            {
                for (int i = 0; i < N; ++i)
                    u(i, k) = singular[k] > cutoff ? b(i, k) / singular[k] : 0;
                singular[k] *= scale;
            }
            struct Singular
            {
                const real *values;
                DNDS_DEVICE_CALLABLE real operator()(int k) const { return values[k]; }
            };
            RefineSVDInverse(a, u, v, Singular{singular}, inverse, c, z, residual, trial, N, steps);
        }
        return true;
    }
}
