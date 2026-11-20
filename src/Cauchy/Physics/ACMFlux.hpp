#pragma once

#include "DNDS/Core/Defines.hpp"

namespace DNDS::Cauchy::ACM
{
    /// Constant-density Scheme A parameters in code units. beta2 has units of
    /// squared velocity. The numerical operator sets roe from its flux policy.
    struct Parameters
    {
        real beta2 = 1, alpha = 0, entropyFix = 0, kinematicViscosity = 0;
        bool roe = false;
    };

    template <class State, class Normal>
    DNDS_DEVICE_CALLABLE inline real NormalSpeed(const State &q, const Normal &n) noexcept
    {
        return q[0] * n[0] + q[1] * n[1] + q[2] * n[2];
    }

    struct Waves
    {
        real minus, tangent, plus;
    };
    DNDS_DEVICE_CALLABLE inline Waves WaveSpeeds(const Parameters &p, real un, real gridNormal = 0) noexcept
    {
        // Eigenvalues of Gamma^-1 (A_n - w_n M), where the physical
        // mass M has a zero pressure row. Only transported rows shift.
        const real half = ((1 - p.alpha) * un - gridNormal) * 0.5;
        const real root = ::hypot(half, ::sqrt(p.beta2));
        // Recover the small root from the product to avoid cancellation.
        const real stable = half >= 0 ? half + root : half - root;
        return half >= 0 ? Waves{-p.beta2 / stable, un - gridNormal, stable}
                         : Waves{stable, un - gridNormal, -p.beta2 / stable};
    }
    DNDS_DEVICE_CALLABLE inline real SpectralRadius(const Parameters &p, real un, real gridNormal = 0) noexcept
    {
        const auto w = WaveSpeeds(p, un, gridNormal);
        return ::fmax(::fabs(w.tangent), ::fmax(-w.minus, w.plus));
    }

    /// Entries of frozen Gamma and dF_n/dQ. Scalar rows advect at u.n.
    template <class State>
    DNDS_DEVICE_CALLABLE inline real GammaEntry(const Parameters &p, const State &q, int row, int col) noexcept
    {
        if (col == 3 && row < 4)
            return (row == 3 ? 1 : (1 + p.alpha) * q[row]) / p.beta2;
        return row == col ? 1 : 0;
    }
    template <class State, class Normal>
    DNDS_DEVICE_CALLABLE inline real FluxJacobianEntry(const State &q, const Normal &n, int row, int col, real gridNormal = 0) noexcept
    {
        if (row == 3)
            return col < 3 ? n[col] : 0;
        return (row == col ? NormalSpeed(q, n) - gridNormal : 0) + (col < 3 ? q[row] * n[col] : 0) +
               (col == 3 && row < 3 ? n[row] : 0);
    }

    struct AbsoluteSpeed
    {
        real delta;
        DNDS_DEVICE_CALLABLE real operator()(real x) const noexcept
        {
            return ::fabs(x) < delta ? 0.5 * (x * (x / delta) + delta) : ::fabs(x);
        }
        DNDS_DEVICE_CALLABLE real Divided(real x, real y) const noexcept
        {
            if (delta > 0 && ::fabs(x) <= delta && ::fabs(y) <= delta)
                return 0.5 * (x + y) / delta;
            if (x >= delta && y >= delta)
                return 1;
            if (x <= -delta && y <= -delta)
                return -1;
            return ((*this)(x) - (*this)(y)) / (x - y);
        }
    };
    struct IncomingWave
    {
        DNDS_DEVICE_CALLABLE real operator()(real x) const noexcept { return x < 0 ? 1 : 0; }
        DNDS_DEVICE_CALLABLE real Divided(real x, real y) const noexcept
        {
            return (x < 0) == (y < 0) ? 0 : ((*this)(x) - (*this)(y)) / (x - y);
        }
    };

    /// Apply f(Gamma^-1 dF_n/dQ) to a jump in global axes. The normal
    /// velocity/pressure 2x2 block has two distinct eigenvalues. Tangential
    /// forcing is -alpha*u_t*delta(u_n). Its divided difference remains valid
    /// when a shear and acoustic speed coincide, without inverting eigenvectors.
    template <class State, class Normal, class Function>
    DNDS_DEVICE_CALLABLE inline void ApplyWaveFunction(const Parameters &p, const State &mean, const Normal &n,
                                                       const real (&jump)[4], Function f, real (&out)[4], real gridNormal = 0) noexcept
    {
        const real un = NormalSpeed(mean, n);
        const auto w = WaveSpeeds(p, un, gridNormal);
        const real dn = NormalSpeed(jump, n);
        const real minus = (jump[3] + w.minus * dn) / (w.minus - w.plus);
        const real plus = dn - minus;
        const real fn = f(w.minus) * minus + f(w.plus) * plus;
        const real coupling = -p.alpha * (f.Divided(w.minus, w.tangent) * minus + f.Divided(w.plus, w.tangent) * plus);
        for (int d = 0; d < 3; ++d)
            out[d] = fn * n[d] + f(w.tangent) * (jump[d] - dn * n[d]) + coupling * (mean[d] - un * n[d]);
        out[3] = -w.plus * f(w.minus) * minus - w.minus * f(w.plus) * plus;
    }

    template <class Left, class Right, class Normal, class Flux>
    DNDS_DEVICE_CALLABLE inline void NumericalFlux(const Parameters &p, const Left &l, const Right &r,
                                                   const Normal &n, Flux &flux, int width = 4, real gridNormal = 0) noexcept
    {
        real mean[4], jump[4], diss[4];
        const real ul = NormalSpeed(l, n), ur = NormalSpeed(r, n);
        for (int j = 0; j < 4; ++j)
        {
            mean[j] = 0.5 * (l[j] + r[j]);
            jump[j] = r[j] - l[j];
        }
        const real radius = SpectralRadius(p, 0.5 * (ul + ur), gridNormal);
        if (p.roe)
            ApplyWaveFunction(p, mean, n, jump, AbsoluteSpeed{p.entropyFix * radius}, diss, gridNormal);
        else
            for (int j = 0; j < 4; ++j)
                diss[j] = radius * jump[j];
        const real dp = diss[3] / p.beta2;
        for (int d = 0; d < 3; ++d)
            flux[d] = 0.5 * ((ul - gridNormal) * l[d] + (ur - gridNormal) * r[d] + (l[3] + r[3]) * n[d] - diss[d] - (1 + p.alpha) * mean[d] * dp);
        flux[3] = 0.5 * (ul + ur - dp);
        const real transport = flux[3] - gridNormal;
        for (int j = 4; j < width; ++j)
            flux[j] = transport * (transport >= 0 ? l[j] : r[j]);
    }
}
