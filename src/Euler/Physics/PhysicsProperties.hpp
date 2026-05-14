/**
 * @file PhysicsProperties.hpp
 * @brief Centralized physics property module — EOS coefficients, transport, kinetics.
 *
 * All interfaces use code-scaled values. Conversions between physical (SI) and
 * code-scaled units happen only inside this module at the Cantera boundary.
 *
 * Scaling conventions:
 *   p_code = p_phys / p0        where p0 = rho0 * U0²
 *   rho_code = rho_phys / rho0
 *   T_code = T_phys / T0
 *   R_code = R_phys / R0        where R0 = U0² / T0   (so R_code = R_phys · T0/U0²)
 *   cp_code = cp_phys / R0      (same scaling as Rgas)
 *   μ_code  = μ_phys / (rho0 · U0)   (L_ref = 1 m)
 *   D_code  = D_phys / U0            (L_ref = 1 m)
 *
 * When T0=rho0=U0=0 (unset), all scaling factors default to 1.
 */

#pragma once

#include "../EulerEvaluatorSettings.hpp"
#include "../Chemistry/ChemicalSource.hpp"

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

        /// Conversion factor R0 = U0² / T0.  R_code = R_phys / R0.
        real invR0() const { return igProp_->U0 * igProp_->U0 / igProp_->T0; }

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

        /// Constant gamma (no state needed) — for initialization / analytic fields.
        real gammaConst() const { return igProp_->gamma; }

        real muRef() const { return igProp_->muGas; }
        real Pr() const { return igProp_->prGas; }
        real TRef() const { return igProp_->TRef; }
        real CSutherland() const { return igProp_->CSutherland; }
        int muModel() const { return igProp_->muModel; }

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
                if (bufY_[k] < 0)
                    bufY_[k] = 0;
                if (bufY_[k] > 1)
                    bufY_[k] = 1;
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
        real muPhys = chemSrc_->viscosity(T, toPhysP(p), massFractions(U));
        return muPhys / (igProp_->rho0 * igProp_->U0);
    }

    template <EulerModel model>
    template <class TU>
    real PhysicsProperties<model>::mixtureConductivity(real T, real p, const TU &U) const
    {
        if (!chemSrc_)
            return Cp(T, U) * mixtureViscosity(T, p, U) / Pr();
        real kPhys = chemSrc_->thermalConductivity(T, toPhysP(p), massFractions(U));
        return kPhys / (igProp_->rho0 * igProp_->U0 * igProp_->U0 * igProp_->U0);
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
        chemSrc_->speciesDiffusivity(T, toPhysP(p), massFractions(U), Dv);
        for (int k = 0; k < Ns; ++k)
            D[k] = Dbuf[k] / igProp_->U0;
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
        chemSrc_->speciesDiffusivity(T, toPhysP(p), massFractions(U), Dv);
        return Dbuf[k] / igProp_->U0;
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
