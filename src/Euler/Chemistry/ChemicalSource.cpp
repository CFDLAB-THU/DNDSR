/**
 * @file ChemicalSource.cpp
 * @brief PIMPL implementation: Cantera is only compiled here.
 */

#include "ChemicalSource.hpp"
#include "DNDS/Errors.hpp"

#include "cantera/core.h"

#include <cmath>
#include <algorithm>

namespace DNDS::Euler::Chemistry
{

    struct ChemicalSource::Impl
    {
        std::shared_ptr<Cantera::Solution> sol;
        Cantera::ThermoPhase *gas = nullptr;
        Cantera::Kinetics *kin = nullptr;
        Cantera::Transport *trn = nullptr;

        int Ns = 0;
        std::vector<std::string> speciesNames;
        std::vector<double> mw;
        std::vector<double> Rk; // species gas constants

        // work buffers
        mutable std::vector<double> bufOmega;
        mutable std::vector<double> bufDwdt, bufDwdp, bufDwdc;
        mutable std::vector<double> bufD;

        void setTPY(double T, double p, ConstSpeciesBufferView Y)
        {
            gas->setMassFractions_NoNorm(Y.data);
            gas->setState_TP(T, p);
        }
    };

    // ---- lifecycle ----------------------------------------------------------

    ChemicalSource::ChemicalSource() : impl_(std::make_unique<Impl>()) {}

    ChemicalSource::ChemicalSource(const std::string &mechanismFile,
                                   const std::string &phaseName)
        : impl_(std::make_unique<Impl>())
    {
        auto &I = *impl_;
        I.sol = Cantera::newSolution(mechanismFile, phaseName, "default");
        I.gas = &(*I.sol->thermo());
        I.kin = &(*I.sol->kinetics());
        I.trn = &(*I.sol->transport());

        I.Ns = static_cast<int>(I.gas->nSpecies());
        I.speciesNames.resize(I.Ns);
        I.mw.resize(I.Ns);
        I.Rk.resize(I.Ns);
        I.gas->getMolecularWeights(I.mw.data());
        for (int k = 0; k < I.Ns; ++k)
        {
            I.speciesNames[k] = I.gas->speciesName(k);
            I.Rk[k] = Cantera::GasConstant / I.mw[k];
        }

        I.bufOmega.resize(I.Ns);
        I.bufDwdt.resize(I.Ns);
        I.bufDwdp.resize(I.Ns);
        I.bufDwdc.resize(I.Ns * I.Ns);
        I.bufD.resize(I.Ns);
    }

    ChemicalSource::~ChemicalSource() = default;

    ChemicalSource::ChemicalSource(ChemicalSource &&) noexcept = default;
    ChemicalSource &ChemicalSource::operator=(ChemicalSource &&) noexcept = default;

    int ChemicalSource::nSpecies() const { return impl_->Ns; }
    int ChemicalSource::nReactions() const { return static_cast<int>(impl_->kin->nReactions()); }
    const std::vector<std::string> &ChemicalSource::speciesNames() const { return impl_->speciesNames; }
    const std::vector<double> &ChemicalSource::molecularWeights() const { return impl_->mw; }

    // ---- mixture properties --------------------------------------------------

    double ChemicalSource::mixtureR(ConstSpeciesBufferView Y) const
    {
        auto &I = *impl_;
        double R = 0;
        for (int k = 0; k < I.Ns; ++k)
            R += Y[k] * I.Rk[k];
        return R;
    }

    double ChemicalSource::mixtureCp(double T, ConstSpeciesBufferView Y) const
    {
        impl_->setTPY(T, 101325, Y);
        return impl_->gas->cp_mass();
    }

    double ChemicalSource::mixtureCv(double T, ConstSpeciesBufferView Y) const
    {
        return mixtureCp(T, Y) - mixtureR(Y);
    }

    double ChemicalSource::mixtureGamma(double T, ConstSpeciesBufferView Y) const
    {
        double cp = mixtureCp(T, Y);
        return cp / std::max(cp - mixtureR(Y), 1e-30);
    }

    double ChemicalSource::speedOfSound(double T, ConstSpeciesBufferView Y) const
    {
        return std::sqrt(mixtureGamma(T, Y) * mixtureR(Y) * T);
    }

    double ChemicalSource::temperatureFromUV(double u, double v,
                                             ConstSpeciesBufferView Y,
                                             double T_guess) const
    {
        impl_->gas->setMassFractions_NoNorm(Y.data);
        double Rmix = mixtureR(Y);
        double T = T_guess > 300 ? T_guess : 300;
        for (int iter = 0; iter < 50; iter++)
        {
            double p = Rmix * T / v;
            impl_->gas->setState_TP(T, p);
            double uCur = impl_->gas->intEnergy_mass();
            double cv = impl_->gas->cv_mass();
            double dT = (u - uCur) / std::max(cv, 1e-6);
            T += dT;
            if (T < 1)
                T = 1;
            if (std::abs(dT) < 1e-6 * std::max(std::abs(T), 1.0))
                break;
        }
        return T;
    }

    // ---- kinetics ------------------------------------------------------------

    void ChemicalSource::productionRates(double T, double p,
                                         ConstSpeciesBufferView Y,
                                         SpeciesBufferView omega) const
    {
        auto &I = *impl_;
        I.setTPY(T, p, Y);
        I.kin->getNetProductionRates(I.bufOmega.data());
        for (int k = 0; k < I.Ns; ++k)
            omega[k] = I.bufOmega[k];
    }

    void ChemicalSource::productionRatesAndJacobian(
        double T, double p, double rho,
        ConstSpeciesBufferView Y,
        SpeciesBufferView omega,
        JacobianBufferView J) const
    {
        auto &I = *impl_;
        I.setTPY(T, p, Y);

        I.kin->getNetProductionRates(I.bufOmega.data());
        for (int k = 0; k < I.Ns; ++k)
            omega[k] = I.bufOmega[k];

        I.kin->getNetProductionRates_ddT(I.bufDwdt.data());
        I.kin->getNetProductionRates_ddP(I.bufDwdp.data());

        // Per-species concentration Jacobian ∂ω_i/∂C_k (sparse Ns×Ns)
        auto dWdC = I.kin->netProductionRates_ddCi();

        // zero Jacobian
        for (int idx = 0; idx < J.rows * J.cols; ++idx)
            J.data[idx] = 0;

        int Ns1 = I.Ns - 1;
        int speciesCol0 = 5;

        // ∂ω/∂(ρY_k) = ∂ω/∂C_k · 1/M_k
        for (int k = 0; k < Ns1; ++k)
        {
            double invMk = 1.0 / std::max(I.mw[k], 1e-30);
            for (int i = 0; i < I.Ns; ++i)
                J(i, speciesCol0 + k) = dWdC.coeff(i, k) * invMk;
        }

        // ∂ω/∂(ρE) = ∂ω/∂T · 1/(ρ·cv)
        double cv = I.gas->cv_mass();
        double dT_drhoe = 1.0 / (rho * std::max(cv, 1e-30));
        for (int i = 0; i < I.Ns; ++i)
            J(i, 4) = I.bufDwdt[i] * dT_drhoe;

        // ∂ω/∂ρ ≈ ∂ω/∂p · ∂p/∂ρ + Σ_k ∂ω/∂C_k · Y_k/M_k
        double cp = I.gas->cp_mass();
        double dp_drho = cp / std::max(cv, 1e-30) * p / std::max(rho, 1e-60);
        for (int i = 0; i < I.Ns; ++i)
        {
            double d = I.bufDwdp[i] * dp_drho;
            for (int kk = 0; kk < I.Ns; ++kk)
                d += dWdC.coeff(i, kk) * Y[kk] / std::max(I.mw[kk], 1e-30);
            J(i, 0) = d;
        }
    }

    // ---- transport -----------------------------------------------------------

    double ChemicalSource::viscosity(double T, double p, ConstSpeciesBufferView Y) const
    {
        impl_->setTPY(T, p, Y);
        return impl_->trn->viscosity();
    }

    double ChemicalSource::thermalConductivity(double T, double p, ConstSpeciesBufferView Y) const
    {
        impl_->setTPY(T, p, Y);
        return impl_->trn->thermalConductivity();
    }

    void ChemicalSource::speciesDiffusivity(double T, double p,
                                            ConstSpeciesBufferView Y,
                                            SpeciesBufferView D) const
    {
        auto &I = *impl_;
        I.setTPY(T, p, Y);
        I.trn->getMixDiffCoeffs(I.bufD.data());
        for (int k = 0; k < I.Ns; ++k)
            D[k] = I.bufD[k];
    }

} // namespace DNDS::Euler::Chemistry
