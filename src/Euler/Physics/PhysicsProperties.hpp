/**
 * @file PhysicsProperties.hpp
 * @brief Centralized physics property module — EOS coefficients, transport, kinetics.
 *
 * Wraps the static IdealGasProperty (always valid, from JSON config) and an
 * optional ChemicalSource (Cantera-backed) for multi-species mixture properties.
 * All accessors fall back to the static values when no ChemicalSource is present.
 *
 * Lives as a member of EulerEvaluator; constructed once from settings.
 */

#pragma once

#include "../EulerEvaluatorSettings.hpp"
#include "../Chemistry/ChemicalSource.hpp"

namespace DNDS::Euler
{

    template <EulerModel model>
    class PhysicsProperties
    {
    public:
        using IdealGas = typename EulerEvaluatorSettings<model>::IdealGasProperty;

        /** @brief Construct from static ideal-gas config. */
        explicit PhysicsProperties(const IdealGas &ig) : igProp_(&ig) {}

        /** @brief Optionally attach a Cantera chemical source for mixture properties. */
        void setChemicalSource(Chemistry::ChemicalSource *src) { chemSrc_ = src; }
        bool hasChemicalSource() const { return chemSrc_ != nullptr; }

        // ---- EOS coefficients (constant — used at init time or without state) ----

        real gamma() const { return igProp_->gamma; }
        real Rgas() const { return igProp_->Rgas; }
        real Cp() const { return igProp_->CpGas; }
        real Cv() const { return igProp_->CpGas - igProp_->Rgas; }
        real muRef() const { return igProp_->muGas; }
        real Pr() const { return igProp_->prGas; }
        real TRef() const { return igProp_->TRef; }
        real CSutherland() const { return igProp_->CSutherland; }
        int muModel() const { return igProp_->muModel; }

        // ---- EOS coefficients (mixture — used with state at face/cell points) ----

        /** Mixture gamma = cp/cv. Falls back to constant igProp if no ChemicalSource. */
        template <class TU>
        real gammaMixture(real T, const TU &U) const
        {
            if (!chemSrc_)
                return igProp_->gamma;
            return chemSrc_->mixtureGamma(T, massFractions(U));
        }

        /** Mixture gas constant R = Σ Y_k R_k. Falls back to constant igProp. */
        template <class TU>
        real RgasMixture(const TU &U) const
        {
            if (!chemSrc_)
                return igProp_->Rgas;
            return chemSrc_->mixtureR(massFractions(U));
        }

        /** Mixture cp. */
        template <class TU>
        real CpMixture(real T, const TU &U) const
        {
            if (!chemSrc_)
                return igProp_->CpGas;
            return chemSrc_->mixtureCp(T, massFractions(U));
        }

        /** Mixture speed of sound: c = sqrt(γ · R · T). */
        template <class TU>
        real speedOfSoundMixture(real T, const TU &U) const
        {
            if (!chemSrc_)
                return std::sqrt(igProp_->gamma * igProp_->Rgas * T);
            return chemSrc_->speedOfSound(T, massFractions(U));
        }

        // ---- Transport (mixture — used at face/cell points) ----

        template <class TU>
        real mixtureViscosity(real T, real p, const TU &U) const
        {
            if (!chemSrc_)
            {
                // Fallback: constant or Sutherland (same as old muEff)
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
            return chemSrc_->viscosity(T, p, massFractions(U));
        }

        template <class TU>
        real mixtureConductivity(real T, real p, const TU &U) const
        {
            if (!chemSrc_)
                return igProp_->CpGas * mixtureViscosity(T, p, U) / igProp_->prGas;
            return chemSrc_->thermalConductivity(T, p, massFractions(U));
        }

        template <class TU, class TD>
        void mixtureDiffusivity(real T, real p, const TU &U, TD &&D) const
        {
            if (!chemSrc_)
            {
                // Fallback: constant Schmidt number Sc = 1.0 → D_k = μ/(ρ·Sc)
                real mu = mixtureViscosity(T, p, U);
                real rho = U[0];
                real D0 = mu / std::max(rho, 1e-60);         // Sc = 1
                int Ns = static_cast<int>(U.size()) - 5 + 1; // nVars = 5 + Ns1 → Ns = nVars - 4
                for (int k = 0; k < Ns; ++k)
                    D[k] = D0;
                return;
            }
            chemSrc_->speciesDiffusivity(T, p, massFractions(U),
                                         Chemistry::SpeciesBufferView{&D.derived()(0),
                                                                      chemSrc_->nSpecies()});
        }

        /**
         * @brief Core species diffusion coefficient D_k [m²/s] at a given state.
         *
         * When ChemicalSource is present, delegates to Cantera mixture-averaged
         * transport. Otherwise uses constant Schmidt number (Sc=1 by default).
         */
        template <class TU>
        real speciesDiffusivityK(real T, real p, const TU &U, int kSpecies) const
        {
            if (!chemSrc_)
            {
                real mu = mixtureViscosity(T, p, U);
                real rho = U[0];
                return mu / std::max(rho, 1e-60); // Sc = 1
            }
            // Single-species diffusivity via Cantera
            int Ns = chemSrc_->nSpecies();
            std::vector<real> Dbuf(Ns);
            Chemistry::SpeciesBufferView Dv{Dbuf.data(), Ns};
            chemSrc_->speciesDiffusivity(T, p, massFractions(U), Dv);
            return Dbuf[kSpecies];
        }

        // ---- Kinetics ----

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
            // Fill persistent buffer on-demand (not thread-safe, but caller serialises)
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
        mutable std::vector<real> bufY_; // scratch for massFractions()
    };

} // namespace DNDS::Euler
