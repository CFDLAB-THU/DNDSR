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
#ifdef DNDS_DIST_MT_USE_OMP
#    include <omp.h>
#endif

namespace DNDS::Euler
{

    template <EulerModel model>
    class PhysicsProperties
    {
    public:
        using Traits = EulerModelTraits<model>;
        using TU = typename Traits::TU;
        using IdealGas = typename EulerEvaluatorSettings<model>::IdealGasProperty;
        using ChemPtr = std::shared_ptr<std::vector<Chemistry::ChemicalSource>>;

        explicit PhysicsProperties(const IdealGas &ig) : igProp_(std::make_unique<const IdealGas>(ig)) {}

        void setChemicalSourcePool(ChemPtr pool) { pool_ = std::move(pool); }
        bool hasChemicalSource() const { return pool_ && pool_->size() > 0; }

        // ---- per-thread chemistry helpers -----------------------------------

    private:
        int threadIdx() const
        {
            DNDS_assert(pool_);
#ifdef DNDS_DIST_MT_USE_OMP
            return omp_get_thread_num() % static_cast<int>(pool_->size());
#else
            return 0;
#endif
        }
        Chemistry::ChemicalSource &chem() const
        {
            int tid = threadIdx();
            DNDS_assert(tid < static_cast<int>(pool_->size()));
            return (*pool_)[tid];
        }

    public:
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

        template <int dim>
        real temperature(const TU &U, real TGuess = 0) const;

        real gamma(real T, const TU &U) const;

        /** Equivalent gamma so that p = (gamma_eq - 1) * rho * e_sensible = rho * Rmix * T.
         *  For non-reactive (constant-gamma) gas, returns the configured gamma.
         *  For reactive gas, computes gamma_eq = 1 + p / (rho * e_sensible)
         *  where p = rho * Rmix(Y) * T (exact ideal-gas EOS) and
         *  e_sensible = (rhoE - KE - rhoH_form) / rho.
         *  @param T  Code-scaled temperature (already from Cantera UV solver).
         *  @param U  Conservative state vector.
         *  @tparam dim Spatial dimension (2 or 3).
         */
        template <int dim>
        real gammaEq(real T, const TU &U) const
        {
            if (!hasChemicalSource())
                return igProp_->gamma;
            DNDS_assert_info(chem().isIdealGas(),
                             "gammaEq(): requires ideal-gas EOS (p = rho·R·T)");
            real rho = U[0];
            real rhoInv = 1.0 / std::max(rho, real(1e-60));
            int I4 = dim + 1;
            real vel2 = 0;
            for (int jd = 1; jd <= dim; ++jd)
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

        real Rgas(const TU &U) const;
        real Cp(real T, const TU &U) const;
        real Cv(real T, const TU &U) const;

        /// Code-scaled volumetric formation enthalpy ρ·Σ Y_k·h_f_k.  0 when no chemistry.
        real mixtureFormationRhoE(const TU &U) const
        {
            if (!hasChemicalSource())
                return 0;
            auto &c = chem();
            int Ns1 = c.nSpecies() - 1;
            int nVars = static_cast<int>(U.size());
            int Isp = nVars - Ns1;
            auto Y = c.massFractions(U[0], &U[Isp], Ns1);
            real invU0sq = 1.0 / (igProp_->U0 * igProp_->U0);
            return c.mixtureFormationRhoE(U[0], Y, invU0sq);
        }

        /** Sensible ρE = total ρE − ρ·Σ Y_k·h_f_k. Returns U[I4] when no chemistry.
         *  @param iEnergy  Index of the energy variable (dim+1). */
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
        real mixtureFormationRhoEIncrement(const TU &dU) const
        {
            if (!hasChemicalSource())
                return 0;
            auto &c = chem();
            int Ns1 = c.nSpecies() - 1;
            int nVars = static_cast<int>(dU.size());
            int Isp = nVars - Ns1;
            double invU0sq = 1.0 / (igProp_->U0 * igProp_->U0);
            c.mixtureFormationRhoESpecies(invU0sq);
            return c.mixtureFormationRhoEIncrement(dU[0], dU.data() + Isp, Ns1);
        }

        /// Constant gamma (no state needed) — for initialization / analytic fields.
        real gammaConst() const { return igProp_->gamma; }

        real muRef() const { return igProp_->muGas; }
        real Pr() const { return igProp_->prGas; }
        real TRef() const { return igProp_->TRef; }
        real CSutherland() const { return igProp_->CSutherland; }
        int muModel() const { return igProp_->muModel; }

        /// Public access to clamped, renormalized mass fractions. Returns a view into per-thread buffer.
        /// Caller must ensure hasChemicalSource() before calling.
        Chemistry::ConstSpeciesBufferView massFractionsPublic(const TU &U) const
        {
            DNDS_assert(hasChemicalSource());
            auto &c = chem();
            int Ns1 = c.nSpecies() - 1;
            int nVars = static_cast<int>(U.size());
            int Isp = nVars - Ns1;
            return c.massFractions(U[0], U.data() + Isp, Ns1);
        }

        // ---- State conversion (I/O helpers, not for tight loops) ------------

        /**
         * @brief cons-total → cons-sensible: subtracts formation enthalpy from U[I4].
         * @note  For I/O purposes — not for performance-critical tight loops.
         */
        template <int dim>
        void consTotalToSensible(const TU &consTotal, TU &consSensible) const
        {
            consSensible = consTotal;
            consSensible[dim + 1] -= mixtureFormationRhoE(consTotal);
        }

        /**
         * @brief cons-sensible → cons-total: adds formation enthalpy to U[I4].
         * @note  For I/O purposes — not for performance-critical tight loops.
         */
        template <int dim>
        void consSensibleToTotal(const TU &consSensible, TU &consTotal) const
        {
            consTotal = consSensible;
            consTotal[dim + 1] += mixtureFormationRhoE(consSensible);
        }

        /**
         * @brief Primitive → conservative.
         *  Iterates gamma via gammaEq for reactive models (cfg.gamma is initial guess).
         *  For non-reactive, uses cfg.gamma directly.
         * @note  For I/O purposes — not for performance-critical tight loops.
         */
        template <int dim>
        void primToConservative(const TU &prim, TU &cons) const
        {
            if (hasChemicalSource())
                DNDS_assert_info(chem().isIdealGas(),
                                 "primToConservative(): requires ideal-gas EOS");

            real rhoH_form = formationFromPrimitive<dim>(prim);
            real gammaUse = igProp_->gamma;
            for (int iter = 0; iter < 10; ++iter)
            {
                Gas::IdealGasThermalPrimitive2Conservative<dim>(prim, cons, gammaUse, rhoH_form);
                if (!hasChemicalSource())
                    break;
                real T_iter = temperature<dim>(cons);
                real gamma_iter = gammaEq<dim>(T_iter, cons);
                if (std::abs(gamma_iter - gammaUse) < 1e-8)
                    break;
                gammaUse = gamma_iter;
            }
        }

        /**
         * @brief Conservative → primitive.
         *  Uses gammaEq from the current conservative state.
         * @note  For I/O purposes — not for performance-critical tight loops.
         */
        template <int dim>
        void conservativeToPrimitive(const TU &cons, TU &prim) const
        {
            if (hasChemicalSource())
                DNDS_assert_info(chem().isIdealGas(),
                                 "conservativeToPrimitive(): requires ideal-gas EOS");

            real T = temperature<dim>(cons);
            real gamma = gammaEq<dim>(T, cons);
            real rhoH_form = mixtureFormationRhoE(cons);
            Gas::IdealGasThermalConservative2Primitive<dim>(cons, prim, gamma, rhoH_form);
        }

        /**
         * @brief prim-rhoT → conservative.  Input: [rho, u, v, (w), T, Y_k].
         * @note  For I/O purposes — not for performance-critical tight loops.
         */
        template <int dim>
        void primRhoTToConservative(const TU &primRhoT, TU &cons) const
        {
            auto prim = primRhoT;
            int I4 = dim + 1;
            real T_code = prim[I4];
            real Rmix = mixtureRfromPrimitive(prim);
            prim[I4] = prim[0] * Rmix * T_code;
            primToConservative<dim>(prim, cons);
        }

        /**
         * @brief Conservative → prim-rhoT.  Output: [rho, u, v, (w), T, Y_k].
         * @note  For I/O purposes — not for performance-critical tight loops.
         */
        template <int dim>
        void conservativeToPrimRhoT(const TU &cons, TU &primRhoT) const
        {
            conservativeToPrimitive<dim>(cons, primRhoT);
            primRhoT[dim + 1] = temperature<dim>(cons);
        }

        /**
         * @brief prim-TP → conservative.  Input: [T, u, v, (w), p, Y_k].
         * @note  For I/O purposes — not for performance-critical tight loops.
         */
        template <int dim>
        void primTPToConservative(const TU &primTP, TU &cons) const
        {
            auto prim = primTP;
            int I4 = dim + 1;
            real T_code = prim[0];
            real p_code = prim[I4];
            real Rmix = mixtureRfromPrimitive(prim);
            prim[0] = p_code / std::max(Rmix * T_code, 1e-60);
            primToConservative<dim>(prim, cons);
        }

        /**
         * @brief Conservative → prim-TP.  Output: [T, u, v, (w), p, Y_k].
         * @note  For I/O purposes — not for performance-critical tight loops.
         */
        template <int dim>
        void conservativeToPrimTP(const TU &cons, TU &primTP) const
        {
            conservativeToPrimitive<dim>(cons, primTP);
            primTP[0] = temperature<dim>(cons);
        }

        // ---- Unit-scaling helpers (I/O only) ---------------------------------

        template <int dim>
        void consCodeToPhys(const TU &code, TU &phys) const
        {
            phys = code;
            phys[0] *= igProp_->rho0;
            for (int j = 1; j <= dim; ++j)
                phys[j] *= (igProp_->rho0 * igProp_->U0);
            phys[dim + 1] *= (igProp_->rho0 * igProp_->U0 * igProp_->U0);
            for (int k = dim + 2; k < (int)phys.size(); ++k)
                phys[k] *= igProp_->rho0;
        }

        template <int dim>
        void consPhysToCode(const TU &phys, TU &code) const
        {
            code = phys;
            code[0] /= igProp_->rho0;
            for (int j = 1; j <= dim; ++j)
                code[j] /= (igProp_->rho0 * igProp_->U0);
            code[dim + 1] /= (igProp_->rho0 * igProp_->U0 * igProp_->U0);
            for (int k = dim + 2; k < (int)code.size(); ++k)
                code[k] /= igProp_->rho0;
        }

        template <int dim>
        void primCodeToPhys(const TU &code, TU &phys) const
        {
            phys = code;
            phys[0] *= igProp_->rho0;
            for (int j = 1; j <= dim; ++j)
                phys[j] *= igProp_->U0;
            phys[dim + 1] *= (igProp_->rho0 * igProp_->U0 * igProp_->U0);
        }

        template <int dim>
        void primPhysToCode(const TU &phys, TU &code) const
        {
            code = phys;
            code[0] /= igProp_->rho0;
            for (int j = 1; j <= dim; ++j)
                code[j] /= igProp_->U0;
            code[dim + 1] /= (igProp_->rho0 * igProp_->U0 * igProp_->U0);
        }

        template <int dim>
        void primRhoTCodeToPhys(const TU &code, TU &phys) const
        {
            phys = code;
            phys[0] *= igProp_->rho0;
            for (int j = 1; j <= dim; ++j)
                phys[j] *= igProp_->U0;
            phys[dim + 1] *= igProp_->T0;
        }

        template <int dim>
        void primRhoTPhysToCode(const TU &phys, TU &code) const
        {
            code = phys;
            code[0] /= igProp_->rho0;
            for (int j = 1; j <= dim; ++j)
                code[j] /= igProp_->U0;
            code[dim + 1] /= igProp_->T0;
        }

        template <int dim>
        void primTPCodeToPhys(const TU &code, TU &phys) const
        {
            phys = code;
            phys[0] *= igProp_->T0;
            for (int j = 1; j <= dim; ++j)
                phys[j] *= igProp_->U0;
            phys[dim + 1] *= (igProp_->rho0 * igProp_->U0 * igProp_->U0);
        }

        template <int dim>
        void primTPPhysToCode(const TU &phys, TU &code) const
        {
            code = phys;
            code[0] /= igProp_->T0;
            for (int j = 1; j <= dim; ++j)
                code[j] /= igProp_->U0;
            code[dim + 1] /= (igProp_->rho0 * igProp_->U0 * igProp_->U0);
        }

    private:
        /// Compute rhoH_form from a primitive vector [rho, u, v, (w), p/T, Y_k].
        template <int dim>
        real formationFromPrimitive(const TU &prim) const
        {
            if (!hasChemicalSource())
                return 0;
            auto &c = chem();
            int Ns1 = c.nSpecies() - 1;
            int nVars = static_cast<int>(prim.size());
            int Isp = nVars - Ns1;
            double invU0sq = 1.0 / (igProp_->U0 * igProp_->U0);
            auto hfView = c.mixtureFormationRhoESpecies(invU0sq);
            real rhoH = 0;
            real sumY = 0;
            for (int k = 0; k < Ns1; ++k)
            {
                rhoH += prim[Isp + k] * prim[0] * hfView[k];
                sumY += prim[Isp + k];
            }
            real lastY = 1.0 - sumY;
            rhoH += lastY * prim[0] * hfView[Ns1];
            return rhoH;
        }

        /// Compute code-scaled mixture Rgas from a primitive vector (uses mass fractions directly).
        real mixtureRfromPrimitive(const TU &prim) const
        {
            if (!hasChemicalSource())
                return toCode(igProp_->Rgas);
            auto &c = chem();
            int Ns1 = c.nSpecies() - 1;
            int nVars = static_cast<int>(prim.size());
            int Isp = nVars - Ns1;
            auto Yview = c.massFractions(1.0, prim.data() + Isp, Ns1); // rho=1 → Y_k directly
            return toCode(c.mixtureR(Yview));
        }

    public:
        // ---- Transport (mixture) --------------------------------------------

        /// Sutherland + const + density-proportional fallback, or Cantera mixture-averaged.
        real mixtureViscosity(real T, real p, const TU &U) const;

        real mixtureConductivity(real T, real p, const TU &U) const;

        template <class TD>
        void mixtureDiffusivity(real T, real p, const TU &U, TD &&D) const;

        real speciesDiffusivityK(real T, real p, const TU &U, int k) const;

        // ---- Kinetics accessor ----------------------------------------------

        int nSpecies() const { return hasChemicalSource() ? chem().nSpecies() : 0; }

        /// Per-species total specific enthalpies in code units (h_k/U0²).
        /// h_k = e_sensible_k + h_f_k + R_k·T  (no KE term).  Sum_k Y_k·h_k = H_mixture.
        void speciesEnthalpies(real T, real p, const TU &U, Chemistry::SpeciesBufferView h) const
        {
            auto &c = chem();
            int Ns1 = c.nSpecies() - 1;
            int nVars = static_cast<int>(U.size());
            int Isp = nVars - Ns1;
            auto Yv = c.massFractions(U[0], U.data() + Isp, Ns1);
            c.speciesEnthalpies(toPhysT(T), toPhysP(p), Yv, h);
            real invU0sq = 1.0 / (igProp_->U0 * igProp_->U0);
            for (int k = 0; k < c.nSpecies(); ++k)
                h[k] *= invU0sq;
        }

        /// Per-species formation-enthalpy in code units (hf_k / U0²) as Eigen Map.
        /// Sum equals mixtureFormationRhoE(U)/rho. Returns empty Map if no chemistry.
        Eigen::Map<const Eigen::Vector<real, Eigen::Dynamic>> mixtureFormationRhoESpecies() const
        {
            if (!hasChemicalSource())
                return {nullptr, 0};
            auto &c = chem();
            double invU0sq = 1.0 / (igProp_->U0 * igProp_->U0);
            auto v = c.mixtureFormationRhoESpecies(invU0sq);
            return Eigen::Map<const Eigen::Vector<real, Eigen::Dynamic>>(v.data, v.nSpecies);
        }

    private:
        Chemistry::ConstSpeciesBufferView massFractions(const TU &U) const
        {
            auto &c = chem();
            int Ns1 = c.nSpecies() - 1;
            int nVars = static_cast<int>(U.size());
            int Isp = nVars - Ns1;
            return c.massFractions(U[0], U.data() + Isp, Ns1);
        }

        ChemPtr pool_;
        std::unique_ptr<const IdealGas> igProp_;
    };

    // ========================================================================
    // Out-of-line template implementations
    // ========================================================================

    template <EulerModel model>
    real PhysicsProperties<model>::mixtureViscosity(real T, real p, const TU &U) const
    {
        if (!hasChemicalSource())
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
        real muPhys = chem().viscosity(toPhysT(T), toPhysP(p), massFractions(U));
        return muPhys / mu0();
    }

    template <EulerModel model>
    real PhysicsProperties<model>::mixtureConductivity(real T, real p, const TU &U) const
    {
        if (!hasChemicalSource())
            return Cp(T, U) * mixtureViscosity(T, p, U) / Pr();
        real kPhys = chem().thermalConductivity(toPhysT(T), toPhysP(p), massFractions(U));
        return kPhys / k0();
    }

    template <EulerModel model>
    template <class TD>
    void PhysicsProperties<model>::mixtureDiffusivity(real T, real p, const TU &U, TD &&D) const
    {
        if (!hasChemicalSource())
            return;
        int Ns = chem().nSpecies();
        std::vector<real> Dbuf(Ns);
        Chemistry::SpeciesBufferView Dv{Dbuf.data(), Ns};
        chem().speciesDiffusivity(toPhysT(T), toPhysP(p), massFractions(U), Dv);
        for (int k = 0; k < Ns; ++k)
            D[k] = Dbuf[k] / D0();
    }

    template <EulerModel model>
    real PhysicsProperties<model>::speciesDiffusivityK(real T, real p, const TU &U, int k) const
    {
        if (!hasChemicalSource())
        {
            real mu = mixtureViscosity(T, p, U);
            return mu / std::max(real(U[0]), 1e-60); // Sc = 1
        }
        int Ns = chem().nSpecies();
        std::vector<real> Dbuf(Ns);
        Chemistry::SpeciesBufferView Dv{Dbuf.data(), Ns};
        chem().speciesDiffusivity(toPhysT(T), toPhysP(p), massFractions(U), Dv);
        return Dbuf[k] / D0();
    }

    // ========================================================================
    // Per-point EOS coefficients — unified constant / mixture dispatch
    // ========================================================================

    template <EulerModel model>
    template <int dim>
    real PhysicsProperties<model>::temperature(const TU &U, real TGuess) const
    {
        real rho = U[0];
        real rhoInv = 1.0 / std::max(rho, 1e-60);
        real vel2 = rhoInv * rhoInv * (U[1] * U[1] + U[2] * U[2]);
        if constexpr (dim == 3)
            vel2 += rhoInv * rhoInv * U[3] * U[3];
        int I4 = dim + 1;
        real uInternal = U[I4] * rhoInv - 0.5 * vel2;
        if (!hasChemicalSource())
        {
            real p = (igProp_->gamma - 1) * rho * uInternal;
            return p * rhoInv / toCode(igProp_->Rgas);
        }
        real uSensible = sensibleRhoE(U, I4) * rhoInv - 0.5 * vel2;
        real uPhys = uInternal * igProp_->U0 * igProp_->U0;
        {
            auto Yv = massFractions(U);
            uPhys -= chem().pVAtReference(Yv); // h_f → u_f(298): subtract (h−u) at T_ref
            if (chem().isIdealGas())
            {
                uPhys -= chem().sensibleInternalEnergyAtReference(Yv);
                // DNDSR stores e_sensible measured from 0 K (ideal-gas convention).
                // Cantera measures thermal energy from T_ref = 298.15 K.
                // For ideal gas: e_sensible(298) = cv(298)·T_ref.
            }
            else
                DNDS_assert_info(false, "temperature(): non-ideal-gas EOS conversion not yet implemented");
        }
        real vPhys = rhoInv / igProp_->rho0;
        double T_guess = TGuess > 0 ? toPhysT(TGuess) : 200.0;
        if (T_guess <= 0)
        {
            real p = (igProp_->gamma - 1) * rho * uSensible;
            T_guess = p * rhoInv / toCode(igProp_->Rgas) * igProp_->T0;
        }
        if (vPhys < 1e-6 || !std::isfinite(vPhys) || !std::isfinite(uPhys))
        {
            DNDS_assert_info(false,
                             fmt::format("temperature(): invalid state vPhys={:.3e} uPhys={:.3e} — "
                                         "non-ideal EOS fallback not implemented",
                                         vPhys, uPhys));
            // Fallback removed: the ideal-gas p/(ρ·Rgas) formula is not valid for
            // non-ideal EOS.  If a recovery path is ever needed, uncomment below:
            // real p = (igProp_->gamma - 1) * rho * uSensible;
            // return p * rhoInv / toCode(igProp_->Rgas);
            return 0;
        }
        double Tphys = chem().temperatureFromUV(uPhys, vPhys, massFractions(U), T_guess);
        return toCodeT(Tphys);
    }

    template <EulerModel model>
    real PhysicsProperties<model>::gamma(real T, const TU &U) const
    {
        if (!hasChemicalSource())
            return igProp_->gamma;
        // Ideal-gas pressure: p = ρ_phys · R_phys · T_phys.  Non-ideal EOS
        // paths would need the actual pressure from the primitive state.
        double pPhys = U(0) * igProp_->rho0 * chem().mixtureR(massFractions(U)) * toPhysT(T);
        return chem().mixtureGamma(toPhysT(T), massFractions(U), pPhys);
    }

    template <EulerModel model>
    real PhysicsProperties<model>::Rgas(const TU &U) const
    {
        if (!hasChemicalSource())
            return toCode(igProp_->Rgas);
        return toCode(chem().mixtureR(massFractions(U)));
    }

    template <EulerModel model>
    real PhysicsProperties<model>::Cp(real T, const TU &U) const
    {
        if (!hasChemicalSource())
            return toCode(igProp_->CpGas);
        double pPhys = U(0) * igProp_->rho0 * chem().mixtureR(massFractions(U)) * toPhysT(T);
        return toCode(chem().mixtureCp(toPhysT(T), massFractions(U), pPhys));
    }

    template <EulerModel model>
    real PhysicsProperties<model>::Cv(real T, const TU &U) const
    {
        if (!hasChemicalSource())
            return Cp(T, U) - Rgas(U);
        double pPhys = U(0) * igProp_->rho0 * chem().mixtureR(massFractions(U)) * toPhysT(T);
        return toCode(chem().mixtureCv(toPhysT(T), massFractions(U), pPhys));
    }

} // namespace DNDS::Euler
