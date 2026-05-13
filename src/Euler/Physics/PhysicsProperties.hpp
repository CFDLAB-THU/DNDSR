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

        // ---- EOS coefficients (constant — used at init time or without state) ----

        real gamma() const { return igProp_->gamma; }
        real Rgas() const { return toCode(igProp_->Rgas); }
        real Cp() const { return toCode(igProp_->CpGas); }
        real Cv() const { return Rgas() != toCode(igProp_->Rgas) ? Cp() - Rgas() : igProp_->CpGas - igProp_->Rgas; }
        real muRef() const { return igProp_->muGas; }
        real Pr() const { return igProp_->prGas; }
        real TRef() const { return igProp_->TRef; }
        real CSutherland() const { return igProp_->CSutherland; }
        int muModel() const { return igProp_->muModel; }

        // ---- EOS coefficients (mixture — used with state at face/cell points) ----

        template <class TU>
        real gammaMixture(real T, const TU &U) const
        {
            return igProp_->gamma;
        } // TODO: wired when γ_mix from Cantera is desired

        template <class TU>
        real RgasMixture(const TU &U) const
        {
            if (!chemSrc_)
                return igProp_->Rgas;
            return toCode(chemSrc_->mixtureR(massFractions(U)));
        }

        template <class TU>
        real CpMixture(real T, const TU &U) const
        {
            if (!chemSrc_)
                return igProp_->CpGas;
            return toCode(chemSrc_->mixtureCp(T, massFractions(U)));
        }

        template <class TU>
        real speedOfSoundMixture(real T, const TU &U) const
        {
            return std::sqrt(gammaMixture(T, U) * RgasMixture(U) * T);
        }

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
            return Cp() * mixtureViscosity(T, p, U) / Pr();
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

} // namespace DNDS::Euler
