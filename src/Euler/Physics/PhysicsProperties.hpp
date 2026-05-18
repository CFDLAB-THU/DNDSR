/**
 * @file PhysicsProperties.hpp
 * @brief Centralized physics property module — EOS coefficients, transport, kinetics.
 *
 * All interfaces use code-scaled values. Conversions between physical (SI) and
 * code-scaled units happen only inside this module at the Cantera boundary.
 *
 * Fundamental reference scales (user-specified):
 *   L0   — reference length  [m]    (default 1)
 *   U0   — reference velocity [m/s]
 *   rho0 — reference density  [kg/m³]
 *   T0   — reference temperature [K] (default 1)
 *
 * Derived scales:
 *   t0   = L0 / U0                       [s]       — time
 *   p0   = rho0 · U0²                    [Pa]      — pressure
 *   R0   = U0² / T0                      [J/(kg·K)] — gas constant / heat capacity
 *   mu0  = rho0 · U0 · L0               [Pa·s]    — dynamic viscosity
 *   k0   = rho0 · U0³ · L0 / T0         [W/(m·K)] — thermal conductivity
 *   D0   = U0 · L0                       [m²/s]    — diffusivity
 *   S0   = rho0 · U0 / L0               [kg/(m³·s)] — volumetric source rate
 *
 * Code-unit conversions:
 *   x_code  = x_phys / x0   for each quantity x.
 */

#pragma once

#include "../EulerEvaluatorSettings.hpp"
#include "../Chemistry/ChemicalSource.hpp"

#include <algorithm>
#include <cmath>

namespace DNDS::Euler
{

    template <EulerModel model>
    class PhysicsProperties
    {
    public:
        using IdealGas = typename EulerEvaluatorSettings<model>::IdealGasProperty;

        explicit PhysicsProperties(const IdealGas &ig) : igProp_(&ig) {}

        void setChemicalSource(Chemistry::ChemicalSource *src) { chemSrc_ = src; }
        bool hasChemicalSource() const { return chemSrc_ != nullptr; }

        // ---- scale helpers --------------------------------------------------

        /// Reference pressure p0 = rho0 · U0².
        real p0() const { return igProp_->rho0 * igProp_->U0 * igProp_->U0; }

        /// Reference time t0 = L0 / U0.
        real t0() const { return igProp_->L0 / igProp_->U0; }

        /// Conversion factor R0 = U0² / T0.  R_code = R_phys / R0.
        real invR0() const { return igProp_->U0 * igProp_->U0 / igProp_->T0; }

        /// Reference dynamic viscosity mu0 = rho0 · U0 · L0.
        real mu0() const { return igProp_->rho0 * igProp_->U0 * igProp_->L0; }

        /// Reference thermal conductivity k0 = rho0 · U0³ · L0 / T0.
        real k0() const { return igProp_->rho0 * igProp_->U0 * igProp_->U0 * igProp_->U0 * igProp_->L0 / igProp_->T0; }

        /// Reference diffusivity D0 = U0 · L0.
        real D0() const { return igProp_->U0 * igProp_->L0; }

        /// Reference volumetric source rate S0 = rho0 · U0 / L0.
        real S0() const { return igProp_->rho0 * igProp_->U0 / igProp_->L0; }

        /// Convert physical gas-constant / heat-capacity to code-scaled:  X_code = X_phys / R0.
        real toCode(real xPhys) const { return xPhys / invR0(); }
        /// Convert code pressure to physical:  p_phys = p_code · p0.
        real toPhysP(real pCode) const { return pCode * p0(); }
        /// Convert code temperature to physical:  T_phys = T_code · T0.
        real toPhysT(real TCode) const { return igProp_->T0 > 0 ? TCode * igProp_->T0 : TCode; }
        /// Convert physical temperature to code-scaled.
        real toCodeT(real TPhys) const { return igProp_->T0 > 0 ? TPhys / igProp_->T0 : TPhys; }

        // ---- EOS coefficients (per-point — uses state T and U vectors) ----

        template <int dim, class TU>
        real temperature(const TU &U, real TGuess = 0) const;

        template <class TU>
        real gamma(real T, const TU &U) const;

        /** Equivalent gamma so that p = (gamma_eq - 1) * rho * e_sensible = rho * Rmix * T.
         *  For non-reactive (constant-gamma) gas, returns the configured gamma.
         *  For reactive gas, computes gamma_eq = 1 + p / (rho * e_sensible)
         *  where p = rho * Rmix(Y) * T (exact ideal-gas EOS) and
         *  e_sensible = (rhoE - KE - rhoH_form) / rho.
         *  @param T  Code-scaled temperature (already from Cantera UV solver).
         *  @param U  Conservative state vector.
         *  @tparam tdim Spatial dimension (2 or 3).
         */
        template <int tdim, class TU>
        real gammaEq(real T, const TU &U) const
        {
            return this->gamma(T, U);
            if (!chemSrc_)
                return igProp_->gamma;
            real rho = U[0];
            real rhoInv = 1.0 / std::max(rho, real(1e-60));
            int I4 = tdim + 1;
            real vel2 = 0;
            for (int jd = 1; jd <= tdim; ++jd)
                vel2 += U[jd] * U[jd];
            vel2 *= rhoInv * rhoInv;
            real rhoH_form = mixtureFormationRhoE(U);
            real e_sensible = (U[I4] - 0.5 * rho * vel2 - rhoH_form) * rhoInv;
            if (e_sensible <= 0)
                return gamma(T, U);        // fallback to cp/cv gamma
            real Rmix = Rgas(U);           // code-scaled
            real p_exact = rho * Rmix * T; // code-scaled

            // std::cout << fmt::format("pTR [{},{},{}], U[{}], gm1[{}], gm1G[{}]", p_exact, T, Rmix, U[4], p_exact / (rho * e_sensible), this->gamma(T, U)) << std::endl;
            return 1.0 + p_exact / (rho * e_sensible);
        }

        template <class TU>
        real Rgas(const TU &U) const;
        template <class TU>
        real Cp(real T, const TU &U) const;
        template <class TU>
        real Cv(real T, const TU &U) const;

        /// Code-scaled volumetric formation enthalpy ρ·Σ Y_k·h_f_k.  0 when no chemistry.
        template <class TU>
        real mixtureFormationRhoE(const TU &U) const
        {
            if (!chemSrc_)
                return 0;
            auto Y = massFractions(U);
            real ePhys = chemSrc_->mixtureFormationEnergy(Y); // J/kg
            real U0sq = igProp_->U0 * igProp_->U0;
            return U[0] * (U0sq > 0 ? ePhys / U0sq : ePhys);
        }

        /** Sensible ρE = total ρE − ρ·Σ Y_k·h_f_k. Returns U[I4] when no chemistry.
         *  @param iEnergy  Index of the energy variable (dim+1). */
        template <class TU>
        real sensibleRhoE(const TU &U, int iEnergy) const
        {
            return U[iEnergy] - mixtureFormationRhoE(U);
        }

        /** Linearized increment of formation enthalpy: d(rhoH_form) from a
         *  conservative-variable increment dU.
         *
         *  d(rhoH_form) = (1/U0²) * Σ_k hf_k · d(ρY_k)
         *
         *  where d(ρY_last) = d(ρ) − Σ_{k<Ns-1} d(ρY_k).  Returns 0 when no
         *  chemistry.  This is the exact differential — no nonlinear terms.
         */
        template <class TU>
        real mixtureFormationRhoEIncrement(const TU &dU) const
        {
            if (!chemSrc_)
                return 0;
            int Ns = chemSrc_->nSpecies();
            int Ns1 = Ns - 1;
            int nVars = static_cast<int>(dU.size());
            int Isp = nVars - Ns1;
            real U0sq = igProp_->U0 * igProp_->U0;
            real scale = (U0sq > 0) ? (1.0 / U0sq) : 1.0;

            if (static_cast<int>(bufHf_.size()) < Ns)
            {
                bufHf_.resize(Ns);
                Chemistry::SpeciesBufferView hfv{bufHf_.data(), Ns};
                chemSrc_->speciesFormationEnthalpies(hfv);
            }

            // sum over independent species
            real dRhoHf = 0;
            real sumDRhoYk = 0;
            for (int k = 0; k < Ns1; ++k)
            {
                dRhoHf += bufHf_[k] * dU[Isp + k];
                sumDRhoYk += dU[Isp + k];
            }
            // last (dependent) species: d(rhoY_last) = d(rho) - sum d(rhoY_k)
            dRhoHf += bufHf_[Ns1] * (dU[0] - sumDRhoYk);
            return dRhoHf * scale;
        }

        /// Constant gamma (no state needed) — for initialization / analytic fields.
        real gammaConst() const { return igProp_->gamma; }

        real muRef() const { return igProp_->muGas; }
        real Pr() const { return igProp_->prGas; }
        real TRef() const { return igProp_->TRef; }
        real CSutherland() const { return igProp_->CSutherland; }
        int muModel() const { return igProp_->muModel; }

        /// Public access to clamped, renormalized mass fractions. Returns a view into internal buffer.
        template <class TU>
        Chemistry::ConstSpeciesBufferView massFractionsPublic(const TU &U) const { return massFractions(U); }

        // ---- Transport (mixture) --------------------------------------------

        /// Sutherland + const + density-proportional fallback, or Cantera mixture-averaged.
        template <class TU>
        real mixtureViscosity(real T, real p, const TU &U) const;

        template <class TU>
        real mixtureConductivity(real T, real p, const TU &U) const;

        template <class TU, class TD>
        void mixtureDiffusivity(real T, real p, const TU &U, TD &&D) const;

        template <class TU>
        real speciesDiffusivityK(real T, real p, const TU &U, int k) const;

        // ---- Kinetics accessor ----------------------------------------------

        Chemistry::ChemicalSource *chemicalSource() { return chemSrc_; }
        const Chemistry::ChemicalSource *chemicalSource() const { return chemSrc_; }

        /// Cached per-species formation enthalpies [J/kg] in physical units.
        /// Populated on first call, returns pointer to Ns elements (or nullptr if no chemistry).
        const real *formationEnthalpies(int &Ns) const
        {
            if (!chemSrc_)
            {
                Ns = 0;
                return nullptr;
            }
            Ns = chemSrc_->nSpecies();
            if (static_cast<int>(bufHf_.size()) < Ns)
            {
                bufHf_.resize(Ns);
                Chemistry::SpeciesBufferView hfv{bufHf_.data(), Ns};
                chemSrc_->speciesFormationEnthalpies(hfv);
            }
            return bufHf_.data();
        }

    private:
        template <class TU>
        Chemistry::ConstSpeciesBufferView massFractions(const TU &U) const
        {
            int Ns = chemSrc_ ? chemSrc_->nSpecies() : 0;
            int Ns1 = Ns - 1;
            int nVars = static_cast<int>(U.size());
            int Isp = nVars - Ns1;
            double rhoInv = 1.0 / std::max(real(U[0]), 1e-60);
            if (static_cast<int>(bufY_.size()) < Ns)
                bufY_.resize(Ns);
            for (int k = 0; k < Ns1; ++k)
                bufY_[k] = U[Isp + k] * rhoInv;
            double sum = 0;
            for (int k = 0; k < Ns1; ++k)
                sum += bufY_[k];
            bufY_[Ns1] = 1.0 - sum;
            for (int k = 0; k < Ns; ++k)
            {
                bufY_[k] = std::max(bufY_[k], real(0));
            }
            real ySum = 0;
            for (int k = 0; k < Ns; ++k)
                ySum += bufY_[k];
            if (ySum > 0)
                for (int k = 0; k < Ns; ++k)
                    bufY_[k] /= ySum;
            return {bufY_.data(), Ns};
        }

        const IdealGas *igProp_ = nullptr;
        Chemistry::ChemicalSource *chemSrc_ = nullptr;
        mutable std::vector<real> bufY_;
        mutable std::vector<real> bufHf_; ///< Cached per-species formation enthalpies [J/kg].
    };

    // ========================================================================
    // Out-of-line template implementations
    // ========================================================================

    template <EulerModel model>
    template <class TU>
    real PhysicsProperties<model>::mixtureViscosity(real T, real p, const TU &U) const
    {
        if (!chemSrc_)
        {
            switch (igProp_->muModel)
            {
            case 0:
                return igProp_->muGas;
            case 1:
            {
                real TRel = T / igProp_->TRef;
                return igProp_->muGas * TRel * std::sqrt(TRel) * (igProp_->TRef + igProp_->CSutherland) / (T + igProp_->CSutherland);
            }
            case 2:
                return igProp_->muGas * U[0];
            default:
                return igProp_->muGas;
            }
        }
        real muPhys = chemSrc_->viscosity(toPhysT(T), toPhysP(p), massFractions(U));
        return muPhys / mu0();
    }

    template <EulerModel model>
    template <class TU>
    real PhysicsProperties<model>::mixtureConductivity(real T, real p, const TU &U) const
    {
        if (!chemSrc_)
            return Cp(T, U) * mixtureViscosity(T, p, U) / Pr();
        real kPhys = chemSrc_->thermalConductivity(toPhysT(T), toPhysP(p), massFractions(U));
        return kPhys / k0();
    }

    template <EulerModel model>
    template <class TU, class TD>
    void PhysicsProperties<model>::mixtureDiffusivity(real T, real p, const TU &U, TD &&D) const
    {
        if (!chemSrc_)
            return;
        int Ns = chemSrc_->nSpecies();
        std::vector<real> Dbuf(Ns);
        Chemistry::SpeciesBufferView Dv{Dbuf.data(), Ns};
        chemSrc_->speciesDiffusivity(toPhysT(T), toPhysP(p), massFractions(U), Dv);
        for (int k = 0; k < Ns; ++k)
            D[k] = Dbuf[k] / D0();
    }

    template <EulerModel model>
    template <class TU>
    real PhysicsProperties<model>::speciesDiffusivityK(real T, real p, const TU &U, int k) const
    {
        if (!chemSrc_)
        {
            real mu = mixtureViscosity(T, p, U);
            return mu / std::max(real(U[0]), 1e-60); // Sc = 1
        }
        int Ns = chemSrc_->nSpecies();
        std::vector<real> Dbuf(Ns);
        Chemistry::SpeciesBufferView Dv{Dbuf.data(), Ns};
        chemSrc_->speciesDiffusivity(toPhysT(T), toPhysP(p), massFractions(U), Dv);
        return Dbuf[k] / D0();
    }

    // ========================================================================
    // Per-point EOS coefficients — unified constant / mixture dispatch
    // ========================================================================

    template <EulerModel model>
    template <int dim, class TU>
    real PhysicsProperties<model>::temperature(const TU &U, real TGuess) const
    {
        real rho = U[0];
        real rhoInv = 1.0 / std::max(rho, 1e-60);
        real vel2 = rhoInv * rhoInv * (U[1] * U[1] + U[2] * U[2]);
        if constexpr (dim == 3)
            vel2 += rhoInv * rhoInv * U[3] * U[3];
        int I4 = dim + 1;
        real uInternal = U[I4] * rhoInv - 0.5 * vel2;
        if (!chemSrc_)
        {
            real p = (igProp_->gamma - 1) * rho * uInternal;
            return p * rhoInv / toCode(igProp_->Rgas);
        }
        real uSensible = sensibleRhoE(U, I4) * rhoInv - 0.5 * vel2;
        real uPhys = uInternal * igProp_->U0 * igProp_->U0;
        real vPhys = rhoInv / igProp_->rho0;
        double T_guess = TGuess > 0 ? toPhysT(TGuess) : 0;
        if (T_guess <= 0)
        {
            real p = (igProp_->gamma - 1) * rho * uSensible;
            T_guess = p * rhoInv / toCode(igProp_->Rgas) * igProp_->T0;
        }
        if (vPhys < 1e-6 || !std::isfinite(vPhys) || !std::isfinite(uPhys))
        {
            static int cnt = 0;
            if (cnt++ < 3)
                fprintf(stderr, "[temp-fb] vPhys=%.3e uPhys=%.3e — using const gamma\n", (double)vPhys, (double)uPhys);
            real p = (igProp_->gamma - 1) * rho * uSensible;
            return p * rhoInv / toCode(igProp_->Rgas);
        }
        double Tphys = chemSrc_->temperatureFromUV(uPhys, vPhys, massFractions(U), T_guess);
        return toCodeT(Tphys);
    }

    template <EulerModel model>
    template <class TU>
    real PhysicsProperties<model>::gamma(real T, const TU &U) const
    {
        if (!chemSrc_)
            return igProp_->gamma;
        return chemSrc_->mixtureGamma(toPhysT(T), massFractions(U));
    }

    template <EulerModel model>
    template <class TU>
    real PhysicsProperties<model>::Rgas(const TU &U) const
    {
        if (!chemSrc_)
            return toCode(igProp_->Rgas);
        return toCode(chemSrc_->mixtureR(massFractions(U)));
    }

    template <EulerModel model>
    template <class TU>
    real PhysicsProperties<model>::Cp(real T, const TU &U) const
    {
        if (!chemSrc_)
            return toCode(igProp_->CpGas);
        return toCode(chemSrc_->mixtureCp(toPhysT(T), massFractions(U)));
    }

    template <EulerModel model>
    template <class TU>
    real PhysicsProperties<model>::Cv(real T, const TU &U) const
    {
        return Cp(T, U) - Rgas(U);
    }

} // namespace DNDS::Euler
