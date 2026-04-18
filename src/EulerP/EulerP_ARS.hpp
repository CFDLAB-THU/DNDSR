#include "EulerP_Physics.hpp"

namespace DNDS::EulerP
{
    DNDS_DEVICE_CALLABLE inline void RoeEigenValueFixer(
        real aL, real aR, real vnL, real vnR,
        real dLambda, real fixScale,
        real &lam0, real &lam123, real &lam4)
    {
        DNDS::IdealGas::EntropyFix_HCorrHY(aL, aR, vnL, vnR, dLambda, fixScale,
                                      lam0, lam123, lam4);
    }

    template <DeviceBackend B, class TUL, class TUR, class TULPrim, class TURPrim>
    DNDS_DEVICE_CALLABLE void RoeAverageNS(TUL &&UL, TUR &&UR,
                                           TULPrim &&ULPrim, TURPrim &&URPrim,
                                           int nVars,
                                           real pL, real pR,
                                           PhysicsDeviceView<B> &phy,
                                           Geom::tPoint &veloRoe,
                                           real &vsqrRoe,
                                           real &HRoe,
                                           real &rhoRoe,
                                           real &aSqrRoe)
    {
        real sqrtRhoLm = std::sqrt(UL(0));
        real sqrtRhoRm = std::sqrt(UR(0));
        real HLm = phy.Pressure2Enthalpy(UL, nVars, pL);
        real HRm = phy.Pressure2Enthalpy(UR, nVars, pR);

        for (int d = 0; d < 3; d++)
            veloRoe(d) = (sqrtRhoLm * ULPrim(d + 1) + sqrtRhoRm * URPrim(d + 1)) / (sqrtRhoLm + sqrtRhoRm);
        vsqrRoe = veloRoe.squaredNorm();
        HRoe = (sqrtRhoLm * HLm + sqrtRhoRm * HRm) / (sqrtRhoLm + sqrtRhoRm);
        // TODO: be more generic here!!! (phy.gamma - 1) is for perfect gas
        aSqrRoe = DNDS::IdealGas::RoeSpeedOfSoundSqr(phy.params.gamma, HRoe, vsqrRoe);
        rhoRoe = sqrtRhoLm * sqrtRhoRm;
    }

    template <DeviceBackend B>
    DNDS_DEVICE_CALLABLE void RoeFluxFlow(const TU &UL, const TU &UR,
                                          real pL, real pR,
                                          const Geom::tPoint &veloRoe,
                                          real vsqrRoe,
                                          real vgn, const Geom::tPoint &n,
                                          real asqrRoe, real aRoe, real HRoe,
                                          PhysicsDeviceView<B> &phy,
                                          real lam0, real lam123, real lam4,
                                          TU &F)
    {
        using TVec = Geom::tPoint;
        TU incU = UR - UL;
        real vnL = U123(UL).dot(n) / UL(0);
        real vnR = U123(UR).dot(n) / UR(0);
        real incU123N = U123(incU).dot(n);
        real veloRoeN = veloRoe.dot(n);

        TVec alpha23V = U123(incU) - incU(0) * veloRoe;
        TVec alpha23VT = alpha23V - n * alpha23V.dot(n);
        real incU4b = incU(I4) - alpha23VT.dot(veloRoe);
        real alpha0, alpha1, alpha4;
        DNDS::IdealGas::RoeAlphaDecomposition(incU(0), incU123N, incU4b,
                                               veloRoeN, HRoe, asqrRoe, aRoe,
                                               phy.params.gamma,
                                               alpha0, alpha1, alpha4);

        alpha0 *= lam0;
        alpha1 *= lam123;
        alpha23VT *= lam123;
        alpha4 *= lam4; // here becomes alpha_i * lam_i

        GasInviscidFlux_XY(UL, nVarsFlow, vnL, vgn, n, pL, F);
        GasInviscidFlux_XY(UR, nVarsFlow, vnR, vgn, n, pR, F);

        F(0) -= alpha0 + alpha1 + alpha4;
        F(I4) -= (HRoe - veloRoeN * aRoe) * alpha0 + 0.5 * vsqrRoe * alpha1 +
                 (HRoe + veloRoeN * aRoe) * alpha4 + alpha23VT.dot(veloRoe);
        for (int d = 0; d < 3; d++)
            F(d + 1) -=
                (veloRoe(d) - aRoe * n(d)) * alpha0 + (veloRoe(d) + aRoe * n(d)) * alpha4 +
                veloRoe(d) * alpha1 + alpha23VT(d);

        for (int i = 0; i < nVarsFlow; i++)
            F(i) *= 0.5;
    }
}