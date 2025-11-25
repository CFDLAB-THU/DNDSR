#include "EulerP_Physics.hpp"

namespace DNDS::EulerP
{
    DNDS_DEVICE_CALLABLE inline void RoeEigenValueFixer(
        real aL, real aR, real vnL, real vnR,
        real dLambda, real fixScale,
        real &lam0, real &lam123, real &lam4)
    {
        const real scaleHartenYee = 0.05 * fixScale;
        const real scaleLD = 0.2 * fixScale;
        const real scaleHFix = 0.25 * fixScale;

        real aAve = 0.5 * (aL + aR);
        real VAve = 0.5 * (vnL + vnR);

        lam0 = std::max(lam0, dLambda * scaleHFix);
        lam4 = std::max(lam4, dLambda * scaleHFix);
        lam123 = std::max(lam123, dLambda * scaleHFix);

        //*HY
        real thresholdHartenYee = scaleHartenYee * (VAve + aAve);
        real thresholdHartenYeeS = sqr(thresholdHartenYee);
        if (lam0 < thresholdHartenYee)
            lam0 = (sqr(lam0) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
        if (lam4 < thresholdHartenYee)
            lam4 = (sqr(lam4) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
        if (lam123 < thresholdHartenYee)
            lam123 = (sqr(lam123) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
        //*HY
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
        aSqrRoe = (phy.params.gamma - 1) * (HRoe - 0.5 * vsqrRoe);
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
        real alpha1 = (phy.params.gamma - 1) / asqrRoe * // TODO: be more generic here!!! (phy.gamma - 1) is for perfect gas
                      (incU(0) * (HRoe - veloRoeN * veloRoeN) +
                       veloRoeN * incU123N - incU4b);
        real alpha0 = (incU(0) * (veloRoeN + aRoe) - incU123N - aRoe * alpha1) / (2 * aRoe);
        real alpha4 = incU(0) - (alpha0 + alpha1);

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