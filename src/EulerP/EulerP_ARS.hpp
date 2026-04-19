/**
 * @file EulerP_ARS.hpp
 * @brief Approximate Riemann Solver (Roe scheme) for the EulerP 5-equation Navier-Stokes system.
 *
 * Provides device-callable functions for:
 * - Entropy-corrected eigenvalue fixing (delegating to @c IdealGas::EntropyFix_HCorrHY)
 * - Roe-averaged state computation (velocity, enthalpy, density, speed of sound)
 * - Complete Roe numerical flux for the 5-equation flow system
 *
 * The Roe flux formula is F = 0.5 * (F_L + F_R - |A_Roe| * (U_R - U_L)),
 * where |A_Roe| is the Roe-averaged absolute flux Jacobian.
 */
#include "EulerP_Physics.hpp"

namespace DNDS::EulerP
{
    /**
     * @brief Applies entropy fix to Roe eigenvalues to prevent expansion shocks.
     *
     * Thin wrapper delegating to @c IdealGas::EntropyFix_HCorrHY, which is shared
     * with the Euler module. Modifies the three characteristic eigenvalues (lam0, lam123, lam4)
     * in-place.
     *
     * @param aL Speed of sound on the left state.
     * @param aR Speed of sound on the right state.
     * @param vnL Normal velocity on the left state.
     * @param vnR Normal velocity on the right state.
     * @param dLambda Entropy fix threshold parameter.
     * @param fixScale Scaling factor for the entropy fix.
     * @param[in,out] lam0 Eigenvalue for the u-a characteristic wave.
     * @param[in,out] lam123 Eigenvalue for the u (contact/shear) waves.
     * @param[in,out] lam4 Eigenvalue for the u+a characteristic wave.
     */
    DNDS_DEVICE_CALLABLE inline void RoeEigenValueFixer(
        real aL, real aR, real vnL, real vnR,
        real dLambda, real fixScale,
        real &lam0, real &lam123, real &lam4)
    {
        DNDS::IdealGas::EntropyFix_HCorrHY(aL, aR, vnL, vnR, dLambda, fixScale,
                                      lam0, lam123, lam4);
    }

    /**
     * @brief Computes Roe-averaged quantities from left and right states.
     *
     * Calculates the Roe-averaged velocity, specific total enthalpy, density, and speed of
     * sound squared using the density-weighted averaging formula:
     * - phi_Roe = (sqrt(rho_L) * phi_L + sqrt(rho_R) * phi_R) / (sqrt(rho_L) + sqrt(rho_R))
     * - rho_Roe = sqrt(rho_L * rho_R)
     *
     * @note Speed of sound squared uses @c IdealGas::RoeSpeedOfSoundSqr, which assumes a perfect gas.
     *
     * @tparam B Device backend (Host or CUDA).
     * @tparam TUL Left conservative state type (deduced).
     * @tparam TUR Right conservative state type (deduced).
     * @tparam TULPrim Left primitive state type (deduced).
     * @tparam TURPrim Right primitive state type (deduced).
     * @param UL Left conservative state vector.
     * @param UR Right conservative state vector.
     * @param ULPrim Left primitive state vector.
     * @param URPrim Right primitive state vector.
     * @param nVars Total number of variables.
     * @param pL Left pressure.
     * @param pR Right pressure.
     * @param phy Physics device view for thermodynamic computations.
     * @param[out] veloRoe Roe-averaged velocity vector (3 components).
     * @param[out] vsqrRoe Roe-averaged velocity magnitude squared.
     * @param[out] HRoe Roe-averaged specific total enthalpy.
     * @param[out] rhoRoe Roe-averaged density.
     * @param[out] aSqrRoe Roe-averaged speed of sound squared.
     */
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

    /**
     * @brief Computes the complete Roe numerical flux for the 5-equation flow system.
     *
     * Implements the Roe approximate Riemann solver:
     * F = 0.5 * (F_L + F_R - |A_Roe| * dU)
     *
     * The procedure:
     * 1. Decomposes the state jump dU = U_R - U_L into Roe characteristic variables (alpha decomposition)
     *    via @c IdealGas::RoeAlphaDecomposition.
     * 2. Scales each alpha by the corresponding entropy-fixed eigenvalue (lam0, lam123, lam4).
     * 3. Accumulates left and right inviscid fluxes via @c GasInviscidFlux_XY.
     * 4. Subtracts the upwind dissipation from the averaged flux.
     * 5. Multiplies the result by 0.5 for the final Roe flux.
     *
     * @tparam B Device backend (Host or CUDA).
     * @param UL Left conservative state.
     * @param UR Right conservative state.
     * @param pL Left pressure.
     * @param pR Right pressure.
     * @param veloRoe Roe-averaged velocity vector (from RoeAverageNS).
     * @param vsqrRoe Roe-averaged velocity magnitude squared.
     * @param vgn Grid normal velocity (for ALE; 0 for static mesh).
     * @param n Outward unit face normal vector.
     * @param asqrRoe Roe-averaged speed of sound squared.
     * @param aRoe Roe-averaged speed of sound.
     * @param HRoe Roe-averaged specific total enthalpy.
     * @param phy Physics device view.
     * @param lam0 Entropy-fixed eigenvalue for the u-a wave.
     * @param lam123 Entropy-fixed eigenvalue for the contact/shear waves.
     * @param lam4 Entropy-fixed eigenvalue for the u+a wave.
     * @param[out] F Output Roe flux vector (must be zero-initialized by caller).
     */
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