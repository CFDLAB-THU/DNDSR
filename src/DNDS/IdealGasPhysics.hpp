#pragma once
/**
 * @file IdealGasPhysics.hpp
 * @brief Shared ideal-gas thermodynamics and Roe-flux primitives.
 *
 * All functions are DNDS_DEVICE_CALLABLE (host + device) and operate on
 * scalar arguments only, so they can be called from both the Eigen-based
 * Euler module and the scalar-loop EulerP module.
 *
 * Primitive variable convention is parameterized via PrimVariable:
 *   - Pressure: prim[I4] stores pressure p  (used by Euler)
 *   - InternalEnergy: prim[I4] stores rho * e_internal (used by EulerP)
 */

#include "DNDS/Defines.hpp"
#include <cmath>

namespace DNDS::IdealGas
{

    /// Which thermodynamic variable is stored at the energy index of the
    /// primitive state vector.
    enum class PrimVariable
    {
        Pressure,       ///< prim[I4] = p  (Euler convention)
        InternalEnergy, ///< prim[I4] = rho * e_internal  (EulerP convention)
    };

    // -----------------------------------------------------------------
    //  Core thermodynamic relations (scalar, device-callable)
    // -----------------------------------------------------------------

    /**
     * @brief Compute pressure, speed-of-sound squared, and specific enthalpy
     *        from total energy, density, and velocity squared.
     */
    DNDS_DEVICE_CALLABLE inline void
    IdealGasThermal(real E, real rho, real vSqr, real gamma,
                    real &p, real &asqr, real &H)
    {
        p = (gamma - 1) * (E - rho * 0.5 * vSqr);
        asqr = gamma * p / rho;
        H = (E + p) / rho;
    }

    /// Pressure from internal energy: p = (gamma - 1) * e
    DNDS_DEVICE_CALLABLE inline real
    Pressure_From_InternalEnergy(real e, real gamma)
    {
        return (gamma - 1) * e;
    }

    /// Internal energy from pressure: e = p / (gamma - 1)
    DNDS_DEVICE_CALLABLE inline real
    InternalEnergy_From_Pressure(real p, real gamma)
    {
        return p / (gamma - 1);
    }

    /// Specific enthalpy from conservative state: H = (E + p) / rho
    DNDS_DEVICE_CALLABLE inline real
    Enthalpy(real E, real rho, real p)
    {
        return (E + p) / rho;
    }

    /// Speed of sound squared: a^2 = gamma * p / rho
    DNDS_DEVICE_CALLABLE inline real
    SpeedOfSoundSqr(real gamma, real p, real rho)
    {
        return gamma * p / rho;
    }

    // -----------------------------------------------------------------
    //  Conservative <-> Primitive conversion (scalar, per-component)
    // -----------------------------------------------------------------

    /**
     * @brief Convert conservative energy to primitive energy-index value.
     *
     * @tparam prim  Whether to store pressure or internal energy.
     * @param E     Total energy (conservative).
     * @param rho   Density.
     * @param vSqr  Velocity squared.
     * @param gamma Ratio of specific heats.
     * @return The value to store at prim[I4].
     */
    template <PrimVariable prim>
    DNDS_DEVICE_CALLABLE inline real
    Cons2PrimEnergy(real E, real rho, real vSqr, real gamma)
    {
        real e = E - rho * 0.5 * vSqr; // rho * e_internal
        if constexpr (prim == PrimVariable::Pressure)
            return (gamma - 1) * e; // p
        else
            return e; // rho * e_internal
    }

    /**
     * @brief Convert primitive energy-index value to conservative total energy.
     *
     * @tparam prim  Whether prim[I4] stores pressure or internal energy.
     * @param primE  The primitive energy-index value (p or e).
     * @param rho    Density.
     * @param vSqr   Velocity squared.
     * @param gamma  Ratio of specific heats.
     * @return Total energy E.
     */
    template <PrimVariable prim>
    DNDS_DEVICE_CALLABLE inline real
    Prim2ConsEnergy(real primE, real rho, real vSqr, real gamma)
    {
        if constexpr (prim == PrimVariable::Pressure)
            return primE / (gamma - 1) + rho * 0.5 * vSqr; // E = p/(gamma-1) + 0.5*rho*v^2
        else
            return primE + rho * 0.5 * vSqr; // E = e + 0.5*rho*v^2
    }

    /**
     * @brief Get pressure from the primitive energy-index value.
     *
     * @tparam prim  Whether prim[I4] stores pressure or internal energy.
     * @param primE  The primitive energy-index value.
     * @param gamma  Ratio of specific heats.
     */
    template <PrimVariable prim>
    DNDS_DEVICE_CALLABLE inline real
    PrimE2Pressure(real primE, real gamma)
    {
        if constexpr (prim == PrimVariable::Pressure)
            return primE;
        else
            return (gamma - 1) * primE;
    }

    // -----------------------------------------------------------------
    //  Roe decomposition primitives (scalar, device-callable)
    // -----------------------------------------------------------------

    /**
     * @brief Roe-averaged speed of sound squared: a^2 = (gamma-1)(H - 0.5*v^2).
     */
    DNDS_DEVICE_CALLABLE inline real
    RoeSpeedOfSoundSqr(real gamma, real HRoe, real vsqrRoe)
    {
        return (gamma - 1) * (HRoe - 0.5 * vsqrRoe);
    }

    /**
     * @brief Roe alpha-decomposition coefficients for the 1D wave structure.
     *
     * Computes the wave strengths alpha0, alpha1, alpha4 from the
     * jump across the interface.
     *
     * @param incU0      Jump in density (UR(0) - UL(0)).
     * @param incU123N   Jump in normal momentum.
     * @param incU4b     Jump in (E - alpha23VT . veloRoe).
     * @param veloRoeN   Roe-averaged normal velocity.
     * @param HRoe       Roe-averaged enthalpy.
     * @param asqrRoe    Roe-averaged speed of sound squared.
     * @param aRoe       Roe-averaged speed of sound.
     * @param gamma      Ratio of specific heats.
     * @param[out] alpha0   Left-going acoustic wave strength.
     * @param[out] alpha1   Entropy wave strength.
     * @param[out] alpha4   Right-going acoustic wave strength.
     */
    DNDS_DEVICE_CALLABLE inline void
    RoeAlphaDecomposition(real incU0, real incU123N, real incU4b,
                          real veloRoeN, real HRoe,
                          real asqrRoe, real aRoe, real gamma,
                          real &alpha0, real &alpha1, real &alpha4)
    {
        alpha1 = (gamma - 1) / asqrRoe *
                 (incU0 * (HRoe - veloRoeN * veloRoeN) +
                  veloRoeN * incU123N - incU4b);
        alpha0 = (incU0 * (veloRoeN + aRoe) - incU123N - aRoe * alpha1) / (2 * aRoe);
        alpha4 = incU0 - (alpha0 + alpha1);
    }

    // -----------------------------------------------------------------
    //  Entropy fix constants (shared between Euler and EulerP)
    // -----------------------------------------------------------------

    static constexpr real kScaleHartenYee = 0.05;
    static constexpr real kScaleLD = 0.2;
    static constexpr real kScaleHFix = 0.25;

    /**
     * @brief H-correction + Harten-Yee entropy fix (scheme 8 in Euler module).
     *
     * This is the entropy fix scheme used by both Euler and EulerP.
     */
    DNDS_DEVICE_CALLABLE inline void
    EntropyFix_HCorrHY(real aL, real aR, real vnL, real vnR,
                        real dLambda, real fixScale,
                        real &lam0, real &lam123, real &lam4)
    {
        const real scaleHartenYee = kScaleHartenYee * fixScale;
        const real scaleHFix = kScaleHFix * fixScale;

        real aAve = 0.5 * (aL + aR);
        real VAve = 0.5 * (vnL + vnR);

        lam0 = std::max(lam0, dLambda * scaleHFix);
        lam4 = std::max(lam4, dLambda * scaleHFix);
        lam123 = std::max(lam123, dLambda * scaleHFix);

        real thresholdHartenYee = scaleHartenYee * (VAve + aAve);
        real thresholdHartenYeeS = thresholdHartenYee * thresholdHartenYee;
        if (lam0 < thresholdHartenYee)
            lam0 = (lam0 * lam0 + thresholdHartenYeeS) / (2 * thresholdHartenYee);
        if (lam4 < thresholdHartenYee)
            lam4 = (lam4 * lam4 + thresholdHartenYeeS) / (2 * thresholdHartenYee);
        if (lam123 < thresholdHartenYee)
            lam123 = (lam123 * lam123 + thresholdHartenYeeS) / (2 * thresholdHartenYee);
    }

} // namespace DNDS::IdealGas
