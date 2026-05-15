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

        std::shared_ptr<Cantera::Solution> solT; // separate phase for temperatureFromUV
        Cantera::ThermoPhase *gasT = nullptr;

        int Ns = 0;
        std::vector<std::string> speciesNames;
        std::vector<double> mw;
        std::vector<double> Rk; // species gas constants
        std::vector<double> hf; // per-species formation enthalpy [J/kg] at 298 K

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

        // Dedicated phase for temperatureFromUV (avoids state corruption from shared phase)
        I.solT = Cantera::newSolution(mechanismFile, phaseName, "");
        I.gasT = &(*I.solT->thermo());

        I.Ns = static_cast<int>(I.gas->nSpecies());
        I.speciesNames.resize(I.Ns);
        I.mw.resize(I.Ns);
        I.Rk.resize(I.Ns);
        I.hf.resize(I.Ns);
        I.gas->getMolecularWeights(I.mw.data());
        for (int k = 0; k < I.Ns; ++k)
        {
            I.speciesNames[k] = I.gas->speciesName(k);
            I.Rk[k] = Cantera::GasConstant / I.mw[k];
            I.hf[k] = I.gas->Hf298SS(k) / I.mw[k];
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
        impl_->gasT->setMassFractions_NoNorm(Y.data);
        double Tinit = T_guess > 300 ? T_guess : 300;
        double p_init = mixtureR(Y) * Tinit / v;
        impl_->gasT->setState_TP(Tinit, p_init);
        impl_->gasT->setState_UV(u, v);
        return impl_->gasT->temperature();
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
        double T, double p, double rho, double rhoE,
        double rhoU, double rhoV, double rhoW,
        int iEnergy, double velScale, double rhoScale,
        ConstSpeciesBufferView Y,
        SpeciesBufferView omega,
        JacobianBufferView J,
        int jacFlags) const
    {
        auto &I = *impl_;
        I.setTPY(T, p, Y);

        I.kin->getNetProductionRates(I.bufOmega.data());
        for (int k = 0; k < I.Ns; ++k)
            omega[k] = I.bufOmega[k];

        I.kin->getNetProductionRates_ddT(I.bufDwdt.data());

        // Per-species concentration Jacobian ∂ω_i/∂C_k (sparse Ns×Ns)
        auto dWdC = I.kin->netProductionRates_ddCi();

        // Per-species h_k/(R_u·T) — needed for u_k = (hRT_k - 1)·R_u·T
        std::vector<double> hRT(I.Ns);
        I.gas->getEnthalpy_RT(hRT.data());

        // zero Jacobian
        for (int idx = 0; idx < J.rows * J.cols; ++idx)
            J.data[idx] = 0;

        int Ns1 = I.Ns - 1;
        int speciesCol0 = iEnergy + 1;

        double cv = I.gas->cv_mass();
        double vs2 = velScale * velScale;
        double cvSafe = std::max(cv, 1e-30);
        double rhoInv = 1.0 / std::max(rho, 1e-60);
        double Ru = Cantera::GasConstant;

        bool skipFluid = jacFlags & JAC_SKIP_FLUID;
        bool skipAbsorb = jacFlags & JAC_SKIP_ABSORPTION;

        int nRows = skipAbsorb ? I.Ns : Ns1; // Ns1 excludes the derived last-species row
        double invMlast = skipAbsorb ? 0.0 : (1.0 / std::max(I.mw[Ns1], 1e-30));

        // ── Species columns (∂ω/∂(ρY_k)_code) ──
        // Concentration chain rule: ∂C_k/∂(ρY_k)_code = rhoScale/MW_k
        // Temperature chain rule:   dT/d(rhoY_k)_code = -(1/(rho_code*cv)) * du_k
        //   (rhoScale cancels: d(rhoY_k)_phys = rhoScale * d(rhoY_k)_code,
        //    but dT/d(rhoY_k)_phys has 1/rho_phys = 1/(rho_code*rhoScale),
        //    so dT/d(rhoY_k)_code = rhoScale * dT/d(rhoY_k)_phys = -du/(rho_code*cv))
        double dT_pre = -rhoInv / cvSafe;
        for (int k = 0; k < Ns1; ++k)
        {
            double invMk = 1.0 / std::max(I.mw[k], 1e-30);
            double du = Ru * T * ((hRT[k] - 1.0) * invMk);
            if (!skipAbsorb)
                du -= Ru * T * ((hRT[Ns1] - 1.0) * invMlast);
            double dT_drY = dT_pre * du;
            for (int i = 0; i < nRows; ++i)
            {
                J(i, speciesCol0 + k) = dWdC.coeff(i, k) * invMk * rhoScale + I.bufDwdt[i] * dT_drY;
                if (!skipAbsorb)
                    J(i, speciesCol0 + k) -= dWdC.coeff(i, Ns1) * invMlast * rhoScale;
            }
        }

        if (skipFluid)
            return;

        // ── Fluid columns (∂ω/∂(ρu_j), ∂ω/∂(ρE), ∂ω/∂ρ) ──

        // ∂ω/∂(ρE)_code = ∂ω/∂T · velScale² / (ρ_code·cv)
        double dT_drhoe = vs2 * rhoInv / cvSafe;
        for (int i = 0; i < nRows; ++i)
            J(i, iEnergy) = I.bufDwdt[i] * dT_drhoe;

        // ∂ω/∂(ρu_j)_code = ∂ω/∂T · dT/d(ρu_j)_code
        double dT_factor = -vs2 * rhoInv * rhoInv / cvSafe;
        for (int jd = 0; jd < iEnergy - 1; ++jd)
        {
            double rhoUk = (jd == 0) ? rhoU : (jd == 1) ? rhoV
                                                        : rhoW;
            if (rhoUk == 0)
                continue;
            double dT_dm = dT_factor * rhoUk;
            for (int i = 0; i < nRows; ++i)
                J(i, 1 + jd) = I.bufDwdt[i] * dT_dm;
        }

        // ∂ω/∂ρ_code = ∂ω/∂T·dT/dρ_code + ∂ω/∂C_last·∂C_last/∂ρ_code
        //   ∂C_last/∂ρ_code = rhoScale/M_last  (since ∂(ρ·Y_last)/∂ρ = 1 at fixed ρY_k)
        double dT_drho = -vs2 * rhoE * rhoInv * rhoInv / cvSafe;
        for (int i = 0; i < nRows; ++i)
        {
            double d = I.bufDwdt[i] * dT_drho;
            if (!skipAbsorb)
                d += dWdC.coeff(i, Ns1) * invMlast * rhoScale;
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

    void ChemicalSource::speciesEnthalpies(double T, double p,
                                           ConstSpeciesBufferView Y,
                                           SpeciesBufferView h) const
    {
        impl_->setTPY(T, p, Y);
        impl_->gas->getPartialMolarEnthalpies(impl_->bufOmega.data());
        for (int k = 0; k < impl_->Ns; ++k)
            h[k] = impl_->bufOmega[k] / std::max(impl_->mw[k], 1e-30);
    }

    void ChemicalSource::speciesFormationEnthalpies(SpeciesBufferView hf) const
    {
        for (int k = 0; k < impl_->Ns; ++k)
            hf[k] = impl_->hf[k];
    }

    double ChemicalSource::mixtureFormationEnergy(ConstSpeciesBufferView Y) const
    {
        double e = 0;
        for (int k = 0; k < impl_->Ns; ++k)
            e += Y[k] * impl_->hf[k];
        return e;
    }

} // namespace DNDS::Euler::Chemistry
