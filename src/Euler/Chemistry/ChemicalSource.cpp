/**
 * @file ChemicalSource.cpp
 * @brief PIMPL implementation: Cantera is only compiled here.
 */

#include "ChemicalSource.hpp"
#include "DNDS/Errors.hpp"

// #undef DNDS_USE_CANTERA
#ifdef DNDS_USE_CANTERA
#    include "cantera/core.h"
#    include "cantera/numerics/Integrator.h"
#    include "cantera/zeroD/IdealGasReactor.h"
#    include "cantera/zeroD/ReactorNet.h"
#else
#    include <Eigen/SparseCore>
#endif

#include <cmath>
#include <algorithm>

namespace DNDS::Euler::Chemistry
{

#ifdef DNDS_USE_CANTERA
    namespace
    {
        class AffineIdealGasConstVolReactor : public Cantera::IdealGasReactor
        {
        public:
            using Cantera::IdealGasReactor::IdealGasReactor;

            void setAffineSpeciesRHS(double chemistryScale, double linearTime,
                                     std::vector<double> constantTerm)
            {
                chemistryScale_ = chemistryScale;
                linearTime_ = linearTime;
                constantTerm_ = std::move(constantTerm);
            }

            void eval(double t, double *LHS, double *RHS) override
            {
                Cantera::IdealGasReactor::eval(t, LHS, RHS);
                DNDS_check_throw_info(constantTerm_.size() == m_nsp,
                                      "AffineIdealGasConstVolReactor: constant term size mismatch");
                DNDS_check_throw_info(linearTime_ > 0,
                                      "AffineIdealGasConstVolReactor: linear time must be positive");

                const double *Y = m_thermo->massFractions();
                const auto &mw = m_thermo->molecularWeights();
                double *mdYdt = RHS + 3;

                double heatRelease = 0.0;
                for (size_t k = 0; k < m_nsp; ++k)
                {
                    double ydotChem = mdYdt[k] / std::max(m_mass, 1e-300);
                    double ydot = chemistryScale_ * ydotChem - Y[k] / linearTime_ + constantTerm_[k];
                    mdYdt[k] = m_mass * ydot;
                    heatRelease -= m_mass * ydot * (m_uk[k] / mw[k]);
                    LHS[k + 3] = m_mass;
                }

                RHS[0] = 0.0;
                RHS[1] = 0.0;
                RHS[2] = heatRelease;
                LHS[2] = m_mass * m_thermo->cv_mass();
            }

        private:
            double chemistryScale_ = 1.0;
            double linearTime_ = 1.0;
            std::vector<double> constantTerm_;
        };

        class ScaledIdealGasConstVolReactor : public Cantera::IdealGasReactor
        {
        public:
            using Cantera::IdealGasReactor::IdealGasReactor;

            void setChemistryScale(double chemistryScale)
            {
                chemistryScale_ = chemistryScale;
            }

            void eval(double t, double *LHS, double *RHS) override
            {
                Cantera::IdealGasReactor::eval(t, LHS, RHS);
                double *mdYdt = RHS + 3;
                for (size_t k = 0; k < m_nsp; ++k)
                    mdYdt[k] *= chemistryScale_;
                RHS[2] *= chemistryScale_;
            }

        private:
            double chemistryScale_ = 1.0;
        };
    }
#endif // DNDS_USE_CANTERA

    struct ChemicalSource::Impl
    {
    private:
#ifdef DNDS_USE_CANTERA
        std::shared_ptr<Cantera::Solution> sol;

        std::shared_ptr<Cantera::Solution> solT; // separate phase for temperatureFromUV

        mutable std::shared_ptr<Cantera::Solution> solCV;
        mutable std::shared_ptr<ScaledIdealGasConstVolReactor> reactorCV;
        mutable std::unique_ptr<Cantera::ReactorNet> netCV;
#endif

    public:
        int Ns = 0;
        std::vector<std::string> speciesNames;
        std::vector<double> mw;
        std::vector<double> Rk; // species gas constants
        std::vector<double> hf; // per-species formation enthalpy [J/kg] at 298 K

        // work buffers
        mutable std::vector<double> bufOmega;
        mutable std::vector<double> bufDwdt, bufDwdp, bufDwdc;
        mutable std::vector<double> bufD;

        Impl(const std::string &mechanismFile, const std::string &phaseName)
        {
            auto &I = *this;
#ifdef DNDS_USE_CANTERA
            I.sol = Cantera::newSolution(mechanismFile, phaseName, "default");
            I.solT = Cantera::newSolution(mechanismFile, phaseName, "");
            auto gas = I.sol->thermo();
            I.Ns = static_cast<int>(gas->nSpecies());
            I.speciesNames.resize(I.Ns);
            I.mw.resize(I.Ns);
            I.Rk.resize(I.Ns);
            I.hf.resize(I.Ns);
            gas->getMolecularWeights(I.mw.data());
            for (int k = 0; k < I.Ns; ++k)
            {
                I.speciesNames[k] = gas->speciesName(k);
                I.Rk[k] = Cantera::GasConstant / I.mw[k];
                I.hf[k] = gas->Hf298SS(k) / I.mw[k];
            }
            I.bufOmega.resize(I.Ns);
            I.bufDwdt.resize(I.Ns);
            I.bufDwdp.resize(I.Ns);
            I.bufDwdc.resize(I.Ns * I.Ns);
            I.bufD.resize(I.Ns);
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
#endif
        }

        double gas_cp_mass() const
        {
#ifdef DNDS_USE_CANTERA
            return sol->thermo()->cp_mass();
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
            return 0.0;
#endif
        }
        double gas_cv_mass() const
        {
#ifdef DNDS_USE_CANTERA
            return sol->thermo()->cv_mass();
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
            return 0.0;
#endif
        }
        double gas_intEnergy_mass() const
        {
#ifdef DNDS_USE_CANTERA
            return sol->thermo()->intEnergy_mass();
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
            return 0.0;
#endif
        }
        double gas_enthalpy_mass() const
        {
#ifdef DNDS_USE_CANTERA
            return sol->thermo()->enthalpy_mass();
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
            return 0.0;
#endif
        }
        double gas_entropy_mass() const
        {
#ifdef DNDS_USE_CANTERA
            return sol->thermo()->entropy_mass();
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
            return 0.0;
#endif
        }
        double gas_soundSpeed() const
        {
#ifdef DNDS_USE_CANTERA
            return sol->thermo()->soundSpeed();
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
            return 0.0;
#endif
        }
        double gas_minTemp() const
        {
#ifdef DNDS_USE_CANTERA
            return sol->thermo()->minTemp();
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
            return 0.0;
#endif
        }
        int kin_nReactions() const
        {
#ifdef DNDS_USE_CANTERA
            auto k = sol->kinetics();
            return static_cast<int>(k->nReactions());
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
            return 0;
#endif
        }
        bool gas_isIdeal() const
        {
#ifdef DNDS_USE_CANTERA
            return sol->thermo()->isIdeal();
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
            return false;
#endif
        }
        double trn_viscosity() const
        {
#ifdef DNDS_USE_CANTERA
            return sol->transport()->viscosity();
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
            return 0.0;
#endif
        }
        double trn_thermalConductivity() const
        {
#ifdef DNDS_USE_CANTERA
            return sol->transport()->thermalConductivity();
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
            return 0.0;
#endif
        }
        void kin_getNetProductionRates(double *omega) const
        {
#ifdef DNDS_USE_CANTERA
            sol->kinetics()->getNetProductionRates(omega);
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
#endif
        }
        void kin_getNetProductionRates_ddT(double *dwdt) const
        {
#ifdef DNDS_USE_CANTERA
            sol->kinetics()->getNetProductionRates_ddT(dwdt);
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
#endif
        }
        void kin_getNetProductionRates_ddP(double *dwdP) const
        {
#ifdef DNDS_USE_CANTERA
            sol->kinetics()->getNetProductionRates_ddP(dwdP);
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
#endif
        }
        auto kin_netProductionRates_ddCi() const
        {
#ifdef DNDS_USE_CANTERA
            return sol->kinetics()->netProductionRates_ddCi();
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
            return Eigen::SparseMatrix<double>();
#endif
        }
        void gas_getPartialMolarEnthalpies(double *u) const
        {
#ifdef DNDS_USE_CANTERA
            sol->thermo()->getPartialMolarEnthalpies(u);
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
#endif
        }
        void gas_getPartialMolarIntEnergies(double *u) const
        {
#ifdef DNDS_USE_CANTERA
            sol->thermo()->getPartialMolarIntEnergies(u);
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
#endif
        }
        void gasT_setMassFractions(const double *Y) const
        {
#ifdef DNDS_USE_CANTERA
            solT->thermo()->setMassFractions_NoNorm(Y);
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
#endif
        }
        void gasT_setState_TP(double T, double p) const
        {
#ifdef DNDS_USE_CANTERA
            solT->thermo()->setState_TP(T, p);
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
#endif
        }
        void gasT_setState_UV(double u, double v, double rtol) const
        {
#ifdef DNDS_USE_CANTERA
            solT->thermo()->setState_UV(u, v, rtol);
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
#endif
        }
        double gasT_temperature() const
        {
#ifdef DNDS_USE_CANTERA
            return solT->thermo()->temperature();
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
            return 0.0;
#endif
        }
        double gasT_cv_mass() const
        {
#ifdef DNDS_USE_CANTERA
            return solT->thermo()->cv_mass();
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
            return 0.0;
#endif
        }
        double gasT_enthalpy_mass() const
        {
#ifdef DNDS_USE_CANTERA
            return solT->thermo()->enthalpy_mass();
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
            return 0.0;
#endif
        }
        double gasT_intEnergy_mass() const
        {
#ifdef DNDS_USE_CANTERA
            return solT->thermo()->intEnergy_mass();
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
            return 0.0;
#endif
        }
        void gasT_getPartialMolarCp(double *cp) const
        {
#ifdef DNDS_USE_CANTERA
            solT->thermo()->getPartialMolarCp(cp);
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
#endif
        }
        void trn_getMixDiffCoeffs(double *d) const
        {
#ifdef DNDS_USE_CANTERA
            sol->transport()->getMixDiffCoeffs(d);
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
#endif
        }

        Impl(const Impl &R)
        {
            auto &c = *this;
#ifdef DNDS_USE_CANTERA
            c.sol = R.sol->clone({}, true, true);
            c.solT = R.solT->clone({}, false, false);
#endif
            c.Ns = R.Ns;
            c.speciesNames = R.speciesNames;
            c.mw = R.mw;
            c.Rk = R.Rk;
            c.hf = R.hf;
            c.bufOmega.resize(Ns);
            c.bufDwdt.resize(Ns);
            c.bufDwdp.resize(Ns);
            c.bufDwdc.resize(Ns * Ns);
            c.bufD.resize(Ns);
        }

        void advanceAffineConstVolume(
            double &T, double rho,
            SpeciesBufferView Y,
            double chemistryScale,
            double linearTime,
            ConstSpeciesBufferView constantTerm,
            double advanceTime,
            double rtol,
            double atol,
            int maxOrder,
            int maxSteps) const
        {
#ifdef DNDS_USE_CANTERA
            DNDS_check_throw_info(Y.nSpecies == this->Ns, "advanceAffineConstVolume(): Y size mismatch");
            DNDS_check_throw_info(constantTerm.nSpecies == this->Ns, "advanceAffineConstVolume(): constant term size mismatch");
            DNDS_check_throw_info(std::isfinite(T) && T > 0, "advanceAffineConstVolume(): T must be positive");
            DNDS_check_throw_info(std::isfinite(rho) && rho > 0, "advanceAffineConstVolume(): rho must be positive");
            DNDS_check_throw_info(std::isfinite(linearTime) && linearTime > 0, "advanceAffineConstVolume(): linearTime must be positive");
            DNDS_check_throw_info(std::isfinite(advanceTime) && advanceTime >= 0, "advanceAffineConstVolume(): advanceTime must be non-negative");

            auto sol = this->sol->clone({}, true, false);
            auto gas = sol->thermo();
            gas->setMassFractions_NoNorm(Y.data);
            gas->setState_TD(T, rho);

            std::vector<double> c(static_cast<size_t>(this->Ns));
            for (int k = 0; k < this->Ns; ++k)
                c[static_cast<size_t>(k)] = constantTerm[k];

            auto reactor = std::make_shared<AffineIdealGasConstVolReactor>(sol, false, "affine_cv");
            reactor->setInitialVolume(1.0 / rho);
            reactor->setChemistryEnabled(true);
            reactor->setEnergyEnabled(true);
            reactor->setAffineSpeciesRHS(chemistryScale, linearTime, std::move(c));

            Cantera::ReactorNet net(reactor);
            net.setTolerances(rtol, atol);
            net.setMaxSteps(maxSteps);
            if (maxOrder > 0)
                net.integrator().setMaxOrder(maxOrder);
            net.advance(advanceTime);

            T = reactor->temperature();
            const double *YEnd = reactor->massFractions();
            for (int k = 0; k < this->Ns; ++k)
                Y[k] = YEnd[k];
#else
            DNDS_assert_info(false, "ChemicalSource::Impl::advanceAffineConstVolume: Cantera not available");
#endif
        }
        void advanceConstVolume(
            double &T, double rho,
            SpeciesBufferView Y,
            double chemistryScale,
            double advanceTime,
            double rtol,
            double atol,
            int maxOrder,
            int maxSteps) const
        {
#ifdef DNDS_USE_CANTERA
            DNDS_check_throw_info(Y.nSpecies == this->Ns, "advanceConstVolume(): Y size mismatch");
            DNDS_check_throw_info(std::isfinite(T) && T > 0, "advanceConstVolume(): T must be positive");
            DNDS_check_throw_info(std::isfinite(rho) && rho > 0, "advanceConstVolume(): rho must be positive");
            DNDS_check_throw_info(std::isfinite(chemistryScale) && chemistryScale >= 0,
                                  "advanceConstVolume(): chemistryScale must be non-negative");
            DNDS_check_throw_info(std::isfinite(advanceTime) && advanceTime >= 0, "advanceConstVolume(): advanceTime must be non-negative");

            auto &I = *this;
            I.ensureConstVolumeReactor();
            I.solCV->thermo()->setMassFractions_NoNorm(Y.data);
            I.solCV->thermo()->setState_TD(T, rho);
            I.reactorCV->setInitialVolume(1.0 / rho);
            I.reactorCV->setChemistryScale(chemistryScale);
            I.reactorCV->syncState();
            I.netCV->setInitialTime(0.0);
            I.netCV->setTolerances(rtol, atol);
            I.netCV->setMaxSteps(maxSteps);
            if (maxOrder > 0)
                I.netCV->integrator().setMaxOrder(maxOrder);
            I.netCV->advance(advanceTime);

            T = I.reactorCV->temperature();
            const double *YEnd = I.reactorCV->massFractions();
            for (int k = 0; k < this->Ns; ++k)
                Y[k] = YEnd[k];
#else
            DNDS_assert_info(false, "ChemicalSource::Impl::advanceConstVolume: Cantera not available");
#endif
        }

        void setTPY(double T, double p, ConstSpeciesBufferView Y) const
        {
#ifdef DNDS_USE_CANTERA
            auto g = sol->thermo();
            g->setMassFractions_NoNorm(Y.data);
            g->setState_TP(T, p);
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
#endif
        }

        void ensureConstVolumeReactor() const
        {
#ifdef DNDS_USE_CANTERA
            if (netCV)
                return;
            solCV = sol->clone({}, true, false);
            reactorCV = std::make_shared<ScaledIdealGasConstVolReactor>(solCV, false, "cv");
            reactorCV->setChemistryEnabled(true);
            reactorCV->setEnergyEnabled(true);
            netCV = std::make_unique<Cantera::ReactorNet>(reactorCV);
#else
            DNDS_assert_info(false, "ChemicalSource::Impl: Cantera not available");
#endif
        }
    };

    // ---- lifecycle ----------------------------------------------------------

    ChemicalSource::ChemicalSource() = default;

    ChemicalSource::ChemicalSource(const std::string &mechanismFile,
                                   const std::string &phaseName)
        : impl_(std::make_unique<Impl>(mechanismFile, phaseName)),
          mechanismFile_(mechanismFile), phaseName_(phaseName)
    {
        DNDS_assert(impl_);
    }

    ChemicalSource::~ChemicalSource() = default;

    ChemicalSource::ChemicalSource(ChemicalSource &&) noexcept = default;
    ChemicalSource &ChemicalSource::operator=(ChemicalSource &&) noexcept = default;

    int ChemicalSource::nSpecies() const
    {
        DNDS_assert(impl_);
        return impl_->Ns;
    }
    int ChemicalSource::nReactions() const
    {
        DNDS_assert(impl_);
        return static_cast<int>(impl_->kin_nReactions());
    }
    const std::vector<std::string> &ChemicalSource::speciesNames() const
    {
        DNDS_assert(impl_);
        return impl_->speciesNames;
    }
    const std::vector<double> &ChemicalSource::molecularWeights() const
    {
        DNDS_assert(impl_);
        return impl_->mw;
    }

    // ---- mixture properties --------------------------------------------------

    double ChemicalSource::mixtureR(ConstSpeciesBufferView Y) const
    {
        DNDS_assert(impl_);
        auto &I = *impl_;
        double R = 0;
        for (int k = 0; k < I.Ns; ++k)
            R += Y[k] * I.Rk[k];
        return R;
    }

    double ChemicalSource::mixtureCp(double T, ConstSpeciesBufferView Y, double p) const
    {
        DNDS_assert(impl_);
        impl_->setTPY(T, p, Y);
        return impl_->gas_cp_mass();
    }

    double ChemicalSource::mixtureCv(double T, ConstSpeciesBufferView Y, double p) const
    {
        DNDS_assert(impl_);
        impl_->setTPY(T, p, Y);
        return impl_->gas_cv_mass();
    }

    double ChemicalSource::mixtureGamma(double T, ConstSpeciesBufferView Y, double p) const
    {
        DNDS_assert(impl_);
        impl_->setTPY(T, p, Y);
        double cp = impl_->gas_cp_mass();
        double cv = impl_->gas_cv_mass();
        return cp / std::max(cv, 1e-30);
    }

    double ChemicalSource::mixtureIntEnergy(double T, ConstSpeciesBufferView Y, double p) const
    {
        DNDS_assert(impl_);
        impl_->setTPY(T, p, Y);
        return impl_->gas_intEnergy_mass();
    }

    double ChemicalSource::mixtureEnthalpy(double T, ConstSpeciesBufferView Y, double p) const
    {
        DNDS_assert(impl_);
        impl_->setTPY(T, p, Y);
        return impl_->gas_enthalpy_mass();
    }

    double ChemicalSource::mixtureEntropy(double T, ConstSpeciesBufferView Y, double p) const
    {
        DNDS_assert(impl_);
        impl_->setTPY(T, p, Y);
        return impl_->gas_entropy_mass();
    }

    double ChemicalSource::minTemperature() const
    {
        DNDS_assert(impl_);
        return impl_->gas_minTemp();
    }

    double ChemicalSource::speedOfSound(double T, ConstSpeciesBufferView Y, double p) const
    {
        DNDS_assert(impl_);
        // Use Cantera's soundSpeed() which computes a^2 = (dp/dρ)_s correctly
        // for both ideal-gas and non-ideal EOS, rather than the manual a = √(γRT).
        impl_->setTPY(T, p, Y);
        return impl_->gas_soundSpeed();
    }

    double ChemicalSource::temperatureFromUV(double u, double v,
                                             ConstSpeciesBufferView Y,
                                             double T_guess,
                                             double rtol) const
    {
        DNDS_assert(impl_);
        impl_->gasT_setMassFractions(Y.data);
        double Tinit = T_guess > 300 ? T_guess : 300;
        double p_init = mixtureR(Y) * Tinit / v;
        impl_->gasT_setState_TP(Tinit, p_init);
        impl_->gasT_setState_UV(u, v, rtol);
        return impl_->gasT_temperature();
    }

    // ---- kinetics ------------------------------------------------------------

    void ChemicalSource::productionRates(double T, double p,
                                         ConstSpeciesBufferView Y,
                                         SpeciesBufferView omega) const
    {
        DNDS_assert(impl_);
        auto &I = *impl_;
        I.setTPY(T, p, Y);
        I.kin_getNetProductionRates(I.bufOmega.data());
        for (int k = 0; k < I.Ns; ++k)
            omega[k] = I.bufOmega[k];
    }

    void ChemicalSource::productionRatesAndJacobian(
        double T, double p, double rho, double rhoE,
        double rhoU, double rhoV, double rhoW,
        int iEnergy, double velScale, double rhoScale,
        ConstSpeciesBufferView Y,
        SpeciesBufferView omega,
        JacobianBufferView dOmegadU,
        int jacFlags) const
    {
        DNDS_assert(impl_);
        auto &J = dOmegadU;
        auto &I = *impl_;
        I.setTPY(T, p, Y);

        I.kin_getNetProductionRates(I.bufOmega.data());
        for (int k = 0; k < I.Ns; ++k)
            omega[k] = I.bufOmega[k];

        I.kin_getNetProductionRates_ddT(I.bufDwdt.data());
        I.kin_getNetProductionRates_ddP(I.bufDwdp.data());

        // Per-species concentration Jacobian ∂ω_i/∂C_k (sparse Ns×Ns)
        auto dWdC = I.kin_netProductionRates_ddCi();

        // Per-species partial molar internal energies u_k [J/kmol] (EOS-agnostic)
        std::vector<double> uBar(I.Ns);
        I.gas_getPartialMolarIntEnergies(uBar.data());

        // DNDSR recovers Cantera temperature from a shifted internal-energy
        // convention: u_cantera = u_DNDSR - pV_ref(Y) - e_sens_ref(Y).  At fixed
        // conservative rhoE, species perturbations therefore change the energy
        // sent to Cantera by the derivative of this reference offset.  For the
        // currently supported ideal-gas bridge, pV_ref + e_sens_ref equals
        // cp_k(T_ref) * T_ref per species mass.
        // TODO(reactive-PP-lower-bound): this bridge is ideal-gas/reference-
        // convention specific.  PP currently assumes a near-zero sensible-energy
        // lower bound, while Cantera mechanisms have a finite valid T range; a
        // general EOS path needs an EOS-aware minimum-energy definition.
        int Ns1 = I.Ns - 1;
        std::vector<double> refOffsetSpecies(I.Ns, 0.0);
        if (I.gas_isIdeal())
        {
            std::vector<double> cpBarRef(I.Ns);
            I.gasT_setMassFractions(Y.data);
            I.gasT_setState_TP(298.15, 101325);
            I.gasT_getPartialMolarCp(cpBarRef.data());
            for (int k = 0; k < I.Ns; ++k)
                refOffsetSpecies[k] = cpBarRef[k] * 298.15 / std::max(I.mw[k], 1e-30);
        }
        std::vector<double> compositionEnergyDiff(Ns1, 0.0);

        // zero Jacobian
        for (int idx = 0; idx < J.rows * J.cols; ++idx)
            J.data[idx] = 0;

        int speciesCol0 = iEnergy + 1;

        double cv = I.gas_cv_mass();
        double vs2 = velScale * velScale;
        double cvSafe = std::max(cv, 1e-30);
        double rhoInv = 1.0 / std::max(rho, 1e-60);

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
        // Pressure chain rule:      dP_phys/d(rhoY_k)_code = (p/T)*dT_drY + rhoScale*T*(Rk - Rlast)
        double PbyT = p / std::max(T, 1e-60);
        double dT_pre = -rhoInv / cvSafe;
        double rhoScaleT = rhoScale * T;
        for (int k = 0; k < Ns1; ++k)
        {
            double invMk = 1.0 / std::max(I.mw[k], 1e-30);
            double du = uBar[k] * invMk; // specific internal energy [J/kg], EOS-agnostic
            if (!skipAbsorb)
            {
                du -= uBar[Ns1] * invMlast;
                du += refOffsetSpecies[k] - refOffsetSpecies[Ns1];
            }
            else
                du += refOffsetSpecies[k];
            compositionEnergyDiff[k] = du;
            double dT_drY = dT_pre * du;
            double dP_drY = PbyT * dT_drY;
            if (!skipAbsorb)
                dP_drY += rhoScaleT * (I.Rk[k] - I.Rk[Ns1]);
            for (int i = 0; i < nRows; ++i)
            {
                J(i, speciesCol0 + k) = dWdC.coeff(i, k) * invMk * rhoScale + I.bufDwdt[i] * dT_drY;
                if (!skipAbsorb)
                    J(i, speciesCol0 + k) -= dWdC.coeff(i, Ns1) * invMlast * rhoScale;
                J(i, speciesCol0 + k) += I.bufDwdp[i] * dP_drY;
            }
        }

        if (skipFluid)
            return;

        // ── Fluid columns (∂ω/∂(ρu_j), ∂ω/∂(ρE), ∂ω/∂ρ) ──

        // ∂ω/∂(ρE)_code = ∂ω/∂T · velScale² / (ρ_code·cv)
        //               + ∂ω/∂P · (p/T) · dT_drhoe
        double dT_drhoe = vs2 * rhoInv / cvSafe;
        for (int i = 0; i < nRows; ++i)
            J(i, iEnergy) = I.bufDwdt[i] * dT_drhoe + I.bufDwdp[i] * PbyT * dT_drhoe;

        // ∂ω/∂(ρu_j)_code = ∂ω/∂T · dT/d(ρu_j)_code + ∂ω/∂P · (p/T)·dT_dm
        double dT_factor = -vs2 * rhoInv * rhoInv / cvSafe;
        for (int jd = 0; jd < iEnergy - 1; ++jd)
        {
            double rhoUk = (jd == 0) ? rhoU : (jd == 1) ? rhoV
                                                        : rhoW;
            if (rhoUk == 0)
                continue;
            double dT_dm = dT_factor * rhoUk;
            for (int i = 0; i < nRows; ++i)
                J(i, 1 + jd) = I.bufDwdt[i] * dT_dm + I.bufDwdp[i] * PbyT * dT_dm;
        }

        // ∂ω/∂ρ_code = ∂ω/∂T·dT/dρ_code + ∂ω/∂C_last·∂C_last/∂ρ_code
        //   ∂C_last/∂ρ_code = rhoScale/M_last  (since ∂(ρ·Y_last)/∂ρ = 1 at fixed ρY_k)
        // At fixed conservative rhoE, momentum, and transported rhoY_k:
        //   u = U0²·(ρE/ρ - |ρu|²/(2ρ²)) - offset(Y)
        // so dT/dρ includes both the total-energy term -U0²·ρE/ρ² and the
        // kinetic correction +U0²·|ρu|²/ρ³, plus the composition/reference-
        // offset contribution from dY_k/dρ = -Y_k/ρ, dY_last/dρ = ΣY_k/ρ.
        double dComposition_drho = 0.0;
        if (!skipAbsorb)
            for (int k = 0; k < Ns1; ++k)
                dComposition_drho += Y[k] * compositionEnergyDiff[k] * rhoInv;
        double rhoMomentum2 = rhoU * rhoU + rhoV * rhoV + rhoW * rhoW;
        double dT_drho = (-vs2 * rhoE * rhoInv * rhoInv +
                          vs2 * rhoMomentum2 * rhoInv * rhoInv * rhoInv +
                          dComposition_drho) /
                         cvSafe;
        for (int i = 0; i < nRows; ++i)
        {
            double d = I.bufDwdt[i] * dT_drho;
            if (!skipAbsorb)
                d += dWdC.coeff(i, Ns1) * invMlast * rhoScale;
            d += I.bufDwdp[i] * PbyT * dT_drho;
            J(i, 0) = d;
        }
    }

    void ChemicalSource::advanceAffineConstVolume(
        double &T, double rho,
        SpeciesBufferView Y,
        double chemistryScale, double linearTime,
        ConstSpeciesBufferView constantTerm,
        double advanceTime, double rtol, double atol,
        int maxOrder, int maxSteps) const
    {
        DNDS_assert(impl_);
        impl_->advanceAffineConstVolume(T, rho, Y, chemistryScale, linearTime,
                                        constantTerm, advanceTime, rtol, atol, maxOrder, maxSteps);
    }

    void ChemicalSource::advanceConstVolume(
        double &T, double rho,
        SpeciesBufferView Y,
        double chemistryScale, double advanceTime,
        double rtol, double atol,
        int maxOrder, int maxSteps) const
    {
        DNDS_assert(impl_);
        impl_->advanceConstVolume(T, rho, Y, chemistryScale, advanceTime,
                                  rtol, atol, maxOrder, maxSteps);
    }

    // ---- transport -----------------------------------------------------------

    double ChemicalSource::viscosity(double T, double p, ConstSpeciesBufferView Y) const
    {
        DNDS_assert(impl_);
        impl_->setTPY(T, p, Y);
        return impl_->trn_viscosity();
    }

    double ChemicalSource::thermalConductivity(double T, double p, ConstSpeciesBufferView Y) const
    {
        DNDS_assert(impl_);
        impl_->setTPY(T, p, Y);
        return impl_->trn_thermalConductivity();
    }

    void ChemicalSource::speciesDiffusivity(double T, double p,
                                            ConstSpeciesBufferView Y,
                                            SpeciesBufferView D) const
    {
        DNDS_assert(impl_);
        auto &I = *impl_;
        I.setTPY(T, p, Y);
        I.trn_getMixDiffCoeffs(I.bufD.data());
        for (int k = 0; k < I.Ns; ++k)
            D[k] = I.bufD[k];
    }

    void ChemicalSource::speciesEnthalpies(double T, double p,
                                           ConstSpeciesBufferView Y,
                                           SpeciesBufferView h) const
    {
        DNDS_assert(impl_);
        impl_->setTPY(T, p, Y);
        impl_->gas_getPartialMolarEnthalpies(impl_->bufOmega.data());
        for (int k = 0; k < impl_->Ns; ++k)
            h[k] = impl_->bufOmega[k] / std::max(impl_->mw[k], 1e-30);
    }

    void ChemicalSource::speciesFormationEnthalpies(SpeciesBufferView hf) const
    {
        DNDS_assert(impl_);
        for (int k = 0; k < impl_->Ns; ++k)
            hf[k] = impl_->hf[k];
    }

    double ChemicalSource::mixtureFormationEnthalpy(ConstSpeciesBufferView Y) const
    {
        DNDS_assert(impl_);
        double e = 0;
        for (int k = 0; k < impl_->Ns; ++k)
            e += Y[k] * impl_->hf[k];
        return e;
    }

    double ChemicalSource::pVAtReference(ConstSpeciesBufferView Y) const
    {
        DNDS_assert(impl_);
        impl_->gasT_setMassFractions(Y.data);
        impl_->gasT_setState_TP(298.15, 101325);
        return impl_->gasT_enthalpy_mass() - impl_->gasT_intEnergy_mass();
    }

    double ChemicalSource::sensibleInternalEnergyAtReference(ConstSpeciesBufferView Y) const
    {
        DNDS_assert(impl_);
        impl_->gasT_setMassFractions(Y.data);
        impl_->gasT_setState_TP(298.15, 101325);
        return impl_->gasT_cv_mass() * 298.15;
    }

    bool ChemicalSource::isIdealGas() const
    {
        DNDS_assert(impl_);
        return impl_->gas_isIdeal();
    }

    // ---- clone ---------------------------------------------------------------

    std::unique_ptr<ChemicalSource> ChemicalSource::clone() const
    {
        DNDS_assert(impl_);
        auto c = std::make_unique<ChemicalSource>();
        c->mechanismFile_ = mechanismFile_;
        c->phaseName_ = phaseName_;
        if (impl_)
            c->impl_ = std::make_unique<Impl>(*impl_);
        return c;
    }

    // ---- per-instance buffers -------------------------------------------------

    ConstSpeciesBufferView ChemicalSource::massFractions(double rho, const double *rhoYK, int nTransported) const
    {
        DNDS_assert(impl_);
        int Ns = impl_->Ns;
        int Ns1 = Ns - 1;
        double rhoInv = 1.0 / std::max(rho, 1e-60);
        if (static_cast<int>(bufY_.size()) < Ns)
            bufY_.resize(Ns);
        for (int k = 0; k < nTransported; ++k)
            bufY_[k] = rhoYK[k] * rhoInv;
        double sum = 0;
        for (int k = 0; k < nTransported; ++k)
            sum += bufY_[k];
        bufY_[Ns1] = 1.0 - sum;
        for (int k = 0; k < Ns; ++k)
            bufY_[k] = std::max(bufY_[k], 0.0);
        double ySum = 0;
        for (int k = 0; k < Ns; ++k)
            ySum += bufY_[k];
        if (ySum > 0)
            for (int k = 0; k < Ns; ++k)
                bufY_[k] /= ySum;
        return {bufY_.data(), Ns};
    }

    ConstSpeciesBufferView ChemicalSource::mixtureFormationRhoESpecies(double invU0sq) const
    {
        DNDS_assert(impl_);
        int Ns = impl_->Ns;
        if (static_cast<int>(bufHf_.size()) < Ns)
        {
            bufHf_.resize(Ns);
            SpeciesBufferView hfv{bufHf_.data(), Ns};
            speciesFormationEnthalpies(hfv);
            for (int k = 0; k < Ns; ++k)
                bufHf_[k] *= invU0sq;
        }
        return {bufHf_.data(), Ns};
    }

    double ChemicalSource::mixtureFormationRhoE(double rho, ConstSpeciesBufferView Y, double invU0sq) const
    {
        return rho * mixtureFormationEnthalpy(Y) * invU0sq;
    }

    double ChemicalSource::mixtureFormationRhoEIncrement(double rhoInc, const double *dRhoYK, int nTransported) const
    {
        DNDS_assert(impl_);
        int Ns = impl_->Ns;
        int Ns1 = Ns - 1;
        // bufHf_ must already be populated with code-scaled values
        const double *hf = bufHf_.data();
        double dRhoHf = hf[Ns1] * rhoInc;
        double sumDRhoYk = 0;
        for (int k = 0; k < nTransported; ++k)
        {
            dRhoHf += hf[k] * dRhoYK[k];
            sumDRhoYk += dRhoYK[k];
        }
        dRhoHf -= hf[Ns1] * sumDRhoYk;
        return dRhoHf;
    }

} // namespace DNDS::Euler::Chemistry
