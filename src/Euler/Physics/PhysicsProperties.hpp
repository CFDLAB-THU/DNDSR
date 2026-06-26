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
 *   rhoU0  = rho0 · U0                    [kg/(m²·s)] — momentum density / mass flux per unit area
 *   rhoE0  = rho0 · U0²                   [Pa = kg/(m·s²)] — total energy density
 *   rhoFlux0  = rho0 · U0                 [kg/(m²·s)] — mass flux per unit face area
 *   rhoUFlux0 = rho0 · U0²                [Pa = kg/(m·s²)] — momentum flux per unit face area
 *   rhoEFlux0 = rho0 · U0³                [kg/s³ = W/m²] — energy flux per unit face area
 *
 * Code-unit conversions:
 *   x_code  = x_phys / x0   for each quantity x.
 */

#pragma once

#include "../EulerEvaluatorSettings.hpp"
#include "../Chemistry/ChemicalSource.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <fmt/ostream.h>
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
        static constexpr int dim = Traits::dim;
        static constexpr auto Seq123 = Traits::Seq123;
        static constexpr auto I4 = dim + 1;

        struct StateConversionOptions
        {
            real temperatureUVTolerance = 1e-12; ///< Cantera setState_UV relative tolerance.
            real gammaTolerance = 1e-8;          ///< primToConservative gamma fixed-point tolerance.
            int gammaMaxIterations = 50;         ///< primToConservative gamma fixed-point cap.
            real totalToStaticTolerance = 1e-10; ///< total-condition static-state fixed-point tolerance.
            int totalToStaticMaxIterations = 60; ///< total-condition bisection cap.
        };

        explicit PhysicsProperties(const IdealGas &ig) : igProp_(std::make_unique<const IdealGas>(ig))
        {
            validateReferenceScales(*igProp_);
        }

        void setChemicalSourcePool(ChemPtr pool) { pool_ = std::move(pool); }
        bool hasChemicalSource() const { return pool_ && pool_->size() > 0; }

        /// Set the runtime RANS model and variable count (for NS_EX with dynamic RANS).
        void setRANS(RANSModel rm)
        {
            ransModel_ = rm;
            switch (rm)
            {
            case RANS_SA:
                nRANS_ = 1;
                break;
            case RANS_KOWilcox:
            case RANS_KOSST:
            case RANS_RKE:
                nRANS_ = 2;
                break;
            default:
                nRANS_ = 0;
                break;
            }
        }

        static constexpr int nRANSVars_static()
        {
            if constexpr (EulerModelTraits<model>::hasSA)
                return 1; // nuTilde
            if constexpr (EulerModelTraits<model>::has2EQ)
                return 2; // k, omega/epsilon
            return 0;
        }
        /// Returns actual RANS variable count: static trait first, then dynamic nRANS_.
        int nRANSVars() const
        {
            int s = nRANSVars_static();
            if (s > 0 || (s == 0 && !EulerModelTraits<model>::isExtended))
                return s;
            return nRANS_;
        }

        // ---- per-thread chemistry helpers -----------------------------------

    private:
        int threadIdx() const
        {
            DNDS_assert(pool_);
#ifdef DNDS_DIST_MT_USE_OMP
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            DNDS_check_throw_info(tid < static_cast<int>(pool_->size()),
                                  fmt::format("ChemicalSource pool has {} entries but OpenMP thread {} requested chemistry",
                                              pool_->size(), tid));
            return tid;
        }

    public:
        [[nodiscard]] Chemistry::ChemicalSource &chem() const
        {
            int tid = threadIdx();
            return (*pool_)[tid];
        }

    public:
        static void validateReferenceScales(const IdealGas &ig)
        {
            DNDS_check_throw_info(std::isfinite(ig.T0) && ig.T0 > 0, "idealGasProperty.T0 must be finite and > 0");
            DNDS_check_throw_info(std::isfinite(ig.rho0) && ig.rho0 > 0, "idealGasProperty.rho0 must be finite and > 0");
            DNDS_check_throw_info(std::isfinite(ig.U0) && ig.U0 > 0, "idealGasProperty.U0 must be finite and > 0");
            DNDS_check_throw_info(std::isfinite(ig.L0) && ig.L0 > 0, "idealGasProperty.L0 must be finite and > 0");
            DNDS_check_throw_info(std::isfinite(ig.gamma) && ig.gamma > 1, "idealGasProperty.gamma must be finite and > 1");
            DNDS_check_throw_info(std::isfinite(ig.Rgas) && ig.Rgas > 0, "idealGasProperty.Rgas must be finite and > 0");
        }

        // ---- scale helpers --------------------------------------------------

        /// Reference density scale
        [[nodiscard]] real rho0() const { return igProp_->rho0; }
        /// Reference speed scale
        [[nodiscard]] real U0() const { return igProp_->U0; }
        /// Reference temperature scale
        [[nodiscard]] real T0() const { return igProp_->T0; }

        /// Reference pressure p0 = rho0 · U0².
        [[nodiscard]] real p0() const { return igProp_->rho0 * igProp_->U0 * igProp_->U0; }

        /// Reference time t0 = L0 / U0.
        [[nodiscard]] real t0() const { return igProp_->L0 / igProp_->U0; }

        /// Conversion factor R0 = U0² / T0.  R_code = R_phys / R0.
        [[nodiscard]] real R0() const { return igProp_->U0 * igProp_->U0 / igProp_->T0; }

        /// Reference dynamic viscosity mu0 = rho0 · U0 · L0.
        [[nodiscard]] real mu0() const { return igProp_->rho0 * igProp_->U0 * igProp_->L0; }

        /// Reference thermal conductivity k0 = rho0 · U0³ · L0 / T0.
        [[nodiscard]] real k0() const { return igProp_->rho0 * igProp_->U0 * igProp_->U0 * igProp_->U0 * igProp_->L0 / igProp_->T0; }

        /// Reference diffusivity D0 = U0 · L0.
        [[nodiscard]] real D0() const { return igProp_->U0 * igProp_->L0; }

        /// Reference volumetric source rate S0 = rho0 · U0 / L0.
        [[nodiscard]] real S0() const { return igProp_->rho0 * igProp_->U0 / igProp_->L0; }

        /// Reference momentum density (mass flux per unit face area) rhoU0 = rho0 · U0.
        [[nodiscard]] real rhoU0() const { return igProp_->rho0 * igProp_->U0; }

        /// Reference total energy density rhoE0 = rho0 · U0².
        [[nodiscard]] real rhoE0() const { return p0(); }

        /// Reference mass flux per unit face area rhoFlux0 = rho0 · U0 (same as rhoU0).
        [[nodiscard]] real rhoFlux0() const { return igProp_->rho0 * igProp_->U0; }

        /// Reference momentum flux per unit face area rhoUFlux0 = rho0 · U0².
        [[nodiscard]] real rhoUFlux0() const { return p0(); }

        /// Reference energy flux per unit face area rhoEFlux0 = rho0 · U0³.
        [[nodiscard]] real rhoEFlux0() const { return igProp_->rho0 * igProp_->U0 * igProp_->U0 * igProp_->U0; }

        /// Convert physical gas-constant / heat-capacity to code-scaled:  X_code = X_phys / R0.
        [[nodiscard]] real toCode(real xPhys) const { return xPhys / R0(); }
        /// Convert code pressure to physical:  p_phys = p_code · p0.
        [[nodiscard]] real toPhysP(real pCode) const { return pCode * p0(); }
        /// Convert code temperature to physical:  T_phys = T_code · T0.
        [[nodiscard]] real toPhysT(real TCode) const { return igProp_->T0 > 0 ? TCode * igProp_->T0 : TCode; }
        /// Convert physical temperature to code-scaled.
        [[nodiscard]] real toCodeT(real TPhys) const { return igProp_->T0 > 0 ? TPhys / igProp_->T0 : TPhys; }

        void resolveStateValue(StateValue &value, int nVars,
                               std::ostream *os = nullptr,
                               const std::string &label = "state") const;

        void printInfo(std::ostream &os) const;

        // ---- EOS coefficients (per-point — uses state T and U vectors) ----

        real temperature(const TU &U, real TGuess = 0, real uvTolerance = 1e-12) const;

        /// Thermodynamic gamma cp/cv used for frozen-composition acoustic speeds.
        real gamma(real T, const TU &U) const;

        /** Equivalent gamma so that p = (gamma_eq - 1) * rho * e_sensible = rho * Rmix * T.
         *  For non-reactive (constant-gamma) gas, returns the configured gamma.
         *  For reactive gas, computes gamma_eq = 1 + p / (rho * e_sensible)
         *  where p = rho * Rmix(Y) * T (exact ideal-gas EOS) and
         *  e_sensible = (rhoE - KE - rhoE_base) / rho.
         *  @param T  Code-scaled temperature (already from Cantera UV solver).
         *  @param U  Conservative state vector.
         *  @tparam dim Spatial dimension (2 or 3).
         */
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
            real rhoE_base = mixtureBaseInternalRhoE(U);
            real e_sensible = (U[I4] - 0.5 * rho * vel2 - rhoE_base) * rhoInv;
            DNDS_assert_info(e_sensible > 0,
                             fmt::format("gammaEq(): e_sensible={:.3e} ≤ 0 — invalid state", e_sensible));
            // NOTE: this assertion fires for zero-temperature + zero-momentum
            // states (e.g. during startup I/O before DOF arrays are initialized).
            // Per the current internal energy convention in the Euler solver,
            // rhoE=0 is the uninitialized sentinel; do not call gammaEq() on it.
            real Rmix = Rgas(U);           // code-scaled
            real p_exact = rho * Rmix * T; // code-scaled

            // std::cout << fmt::format("pTR [{},{},{}], U[{}], gm1[{}], gm1G[{}]", p_exact, T, Rmix, U[4], p_exact / (rho * e_sensible), this->gamma(T, U)) << std::endl;
            return 1.0 + p_exact / (rho * e_sensible);
        }

        real Rgas(const TU &U) const;
        real Cp(real T, const TU &U) const;
        real Cv(real T, const TU &U) const;

        struct conservativeThermalReturn
        {
            real T = UnInitReal;
            real p = UnInitReal;
            real asqr = UnInitReal;
            real H = UnInitReal;
            /// p = (gammaEq - 1) * rho * e_sensible = rho * Rmix * T
            real gammaEq = UnInitReal;
            /// Thermodynamic gamma cp/cv, used for frozen-composition acoustic speed a² = gamma·p/ρ
            real gamma = UnInitReal;
        };

        /// Compute temperature, pressure, sound speed, enthalpy, and both gamma values.
        /// @param TGuess warm-start for temperature-from-internal-energy solve
        /// @param uvTolerance convergence tolerance for temperature solve
        template <class TVar>
        [[nodiscard]] conservativeThermalReturn conservativeThermal(TVar &&U, real TGuess = 0, real uvTolerance = 1e-8) const
        {
            real T = this->temperature(U, TGuess, uvTolerance);
            real asqr = UnInitReal;
            real p = UnInitReal;
            real H = UnInitReal;
            real gammaEqUse = gammaEq(T, U);
            real gammaUse = gamma(T, U); // cp/cv, used for frozen acoustic speed
            Gas::IdealGasThermal(U(I4), U(0), (U(Seq123) / U(0)).squaredNorm(),
                                 gammaEqUse, gammaUse, p, asqr, H,
                                 mixtureBaseInternalRhoE(U));
            return {T, p, asqr, H, gammaEqUse, gammaUse};
        }

        /// Code-scaled volumetric base internal energy ρ·Σ Y_k·e_base,k.  0 when no chemistry.
        real mixtureBaseInternalRhoE(const TU &U) const
        {
            if (!hasChemicalSource())
                return 0;
            auto &c = chem();
            int Ns1 = c.nSpecies() - 1;
            int nVars = static_cast<int>(U.size());
            int Isp = nVars - Ns1;
            DNDS_check_throw_info(Isp >= 1, "mixtureBaseInternalRhoE(): state vector too small for mechanism species count");
            auto Y = massFractionsVector(U);
            return c.mixtureBaseInternalRhoE(U[0], {Y.data(), static_cast<int>(Y.size())});
        }

        /// Raw linear base internal energy from rho/rhoY without clipping or renormalizing species.
        /// Intentionally permits negative independent/dependent species masses (e.g. sum(rhoY_k) > rho)
        /// in order to preserve exact linearity through reconstruction and compression algebra.
        /// Callers are responsible for downstream species-positivity enforcement
        /// (checkRecBaseGood, CompressRecPart, CompressInc, AddFixedIncrement).
        real mixtureBaseInternalRhoERaw(const TU &U) const
        {
            if (!hasChemicalSource())
                return 0;
            auto &c = chem();
            int Ns1 = c.nSpecies() - 1;
            int nVars = static_cast<int>(U.size());
            int Isp = nVars - Ns1;
            DNDS_check_throw_info(Isp >= 1, "mixtureBaseInternalRhoERaw(): state vector too small for mechanism species count");
            auto eBaseView = c.mixtureBaseInternalRhoESpecies();
            real rhoEBase = 0;
            real sumRhoY = 0;
            for (int k = 0; k < Ns1; ++k)
            {
                rhoEBase += U[Isp + k] * eBaseView[k];
                sumRhoY += U[Isp + k];
            }
            rhoEBase += (U[0] - sumRhoY) * eBaseView[Ns1];
            return rhoEBase;
        }

        /** Sensible ρE = total ρE − ρ·Σ Y_k·e_base,k. Returns U[I4] when no chemistry.
         *  @param iEnergy  Index of the energy variable (dim+1). */
        real sensibleRhoE(const TU &U, int iEnergy) const
        {
            return U[iEnergy] - mixtureBaseInternalRhoE(U);
        }

        /** Linearized increment of base internal energy: d(rhoE_base) from a
         *  conservative-variable increment dU.
         *
         *  d(rhoE_base) = (1/U0²) * Σ_k e_base,k · d(ρY_k)
         *
         *  where d(ρY_last) = d(ρ) − Σ_{k<Ns-1} d(ρY_k).  Returns 0 when no
         *  chemistry.  This is the exact differential — no nonlinear terms.
         */
        real mixtureBaseInternalRhoEIncrement(const TU &dU) const
        {
            if (!hasChemicalSource())
                return 0;
            auto &c = chem();
            int Ns1 = c.nSpecies() - 1;
            int nVars = static_cast<int>(dU.size());
            int Isp = nVars - Ns1;
            DNDS_check_throw_info(Isp >= 1, "mixtureBaseInternalRhoEIncrement(): state vector too small for mechanism species count");
            return c.mixtureBaseInternalRhoEIncrement(dU[0], dU.data() + Isp, Ns1);
        }

        /// Constant gamma (no state needed) — for initialization / analytic fields.
        [[nodiscard]] real gammaConst() const { return igProp_->gamma; }

        [[nodiscard]] real muRef() const { return igProp_->muGas; }
        [[nodiscard]] real Pr() const { return igProp_->prGas; }
        [[nodiscard]] real TRef() const { return igProp_->TRef; }
        [[nodiscard]] real CSutherland() const { return igProp_->CSutherland; }
        [[nodiscard]] int muModel() const { return igProp_->muModel; }

        /// Public access to clamped, renormalized mass fractions. Caller owns @p Y storage.
        /// Caller must ensure hasChemicalSource() before calling.
        void massFractionsPublic(const TU &U, Chemistry::SpeciesBufferView Y) const
        {
            DNDS_assert(hasChemicalSource());
            auto &c = chem();
            int Ns1 = c.nSpecies() - 1;
            int nVars = static_cast<int>(U.size());
            int Isp = nVars - Ns1;
            c.massFractions(U[0], U.data() + Isp, Ns1, Y);
        }

        // ---- State conversion (I/O helpers, not for tight loops) ------------

        /**
         * @brief cons-total → cons-sensible: subtracts base internal energy from U[I4].
         * @note  For I/O purposes — not for performance-critical tight loops.
         */
        void consTotalToSensible(const TU &consTotal, TU &consSensible) const
        {
            consSensible = consTotal;
            consSensible[dim + 1] -= mixtureBaseInternalRhoE(consTotal);
        }

        /**
         * @brief cons-sensible → cons-total: adds base internal energy to U[I4].
         * @note  For I/O purposes — not for performance-critical tight loops.
         */
        void consSensibleToTotal(const TU &consSensible, TU &consTotal) const
        {
            consTotal = consSensible;
            consTotal[dim + 1] += mixtureBaseInternalRhoE(consSensible);
        }

        /**
         * @brief Primitive → conservative.
         *  For reactive ideal gases, builds total energy directly from Cantera
         *  internal energy.  For non-reactive gas, uses cfg.gamma directly.
         * @note  For I/O purposes — not for performance-critical tight loops.
         */
        void primToConservative(const TU &prim, TU &cons,
                                const StateConversionOptions &options = StateConversionOptions{}) const
        {
            if (hasChemicalSource())
                DNDS_assert_info(chem().isIdealGas(),
                                 "primToConservative(): requires ideal-gas EOS");

            TU primUse = sanitizePrimitiveSpecies(prim);
            validatePrimitiveRhoP(primUse, "primToConservative");
            real gammaEqUse = igProp_->gamma;
            if (!hasChemicalSource())
            {
                Gas::IdealGasThermalPrimitive2Conservative<dim>(primUse, cons, gammaEqUse);
                return;
            }

            auto &c = chem();
            int Ns1 = c.nSpecies() - 1;
            int nVars = static_cast<int>(primUse.size());
            int Isp = nVars - Ns1;
            int I4 = dim + 1;
            std::vector<double> Ybuf(static_cast<size_t>(c.nSpecies()));
            Chemistry::SpeciesBufferView Y{Ybuf.data(), c.nSpecies()};
            c.massFractions(1.0, primUse.data() + Isp, Ns1, Y);
            real Rmix = mixtureRfromPrimitive(primUse);
            real T_code = primUse[I4] / std::max(primUse[0] * Rmix, real(1e-60));
            double uPhys = c.mixtureIntEnergy(toPhysT(T_code), Y, toPhysP(primUse[I4]));

            cons = primUse * primUse[0];
            real vel2 = primUse(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).squaredNorm();
            cons[0] = primUse[0];
            cons[I4] = primUse[0] * uPhys / (igProp_->U0 * igProp_->U0) + 0.5 * primUse[0] * vel2;
        }

        /**
         * @brief Conservative → primitive.
         *  Uses gammaEq from the current conservative state.
         * @note  For I/O purposes — not for performance-critical tight loops.
         */
        void conservativeToPrimitive(const TU &cons, TU &prim,
                                     const StateConversionOptions &options = StateConversionOptions{}) const
        {
            if (hasChemicalSource())
                DNDS_assert_info(chem().isIdealGas(),
                                 "conservativeToPrimitive(): requires ideal-gas EOS");

            real T = temperature(cons, 0, options.temperatureUVTolerance);
            real gammaEqUse = gammaEq(T, cons);
            real rhoE_base = mixtureBaseInternalRhoE(cons);
            Gas::IdealGasThermalConservative2Primitive<dim>(cons, prim, gammaEqUse, rhoE_base);
        }

        /**
         * @brief prim-rhoT → conservative.  Input: [rho, u, v, (w), T, Y_k].
         * @note  For I/O purposes — not for performance-critical tight loops.
         */
        void primRhoTToConservative(const TU &primRhoT, TU &cons,
                                    const StateConversionOptions &options = StateConversionOptions{}) const
        {
            auto prim = sanitizePrimitiveSpecies(primRhoT);
            int I4 = dim + 1;
            real T_code = prim[I4];
            DNDS_check_throw_info(prim[0] > 0 && T_code > 0,
                                  "primRhoTToConservative(): rho and T must be positive");
            real Rmix = mixtureRfromPrimitive(prim);
            prim[I4] = prim[0] * Rmix * T_code;
            primToConservative(prim, cons, options);
        }

        /**
         * @brief Conservative → prim-rhoT.  Output: [rho, u, v, (w), T, Y_k].
         * @note  For I/O purposes — not for performance-critical tight loops.
         */
        void conservativeToPrimRhoT(const TU &cons, TU &primRhoT,
                                    const StateConversionOptions &options = StateConversionOptions{}) const
        {
            conservativeToPrimitive(cons, primRhoT, options);
            primRhoT[dim + 1] = temperature(cons, 0, options.temperatureUVTolerance);
        }

        /**
         * @brief prim-TP → conservative.  Input: [T, u, v, (w), p, Y_k].
         * @note  For I/O purposes — not for performance-critical tight loops.
         */
        void primTPToConservative(const TU &primTP, TU &cons,
                                  const StateConversionOptions &options = StateConversionOptions{}) const
        {
            auto prim = sanitizePrimitiveSpecies(primTP);
            int I4 = dim + 1;
            real T_code = prim[0];
            real p_code = prim[I4];
            DNDS_check_throw_info(T_code > 0 && p_code > 0,
                                  "primTPToConservative(): T and p must be positive");
            real Rmix = mixtureRfromPrimitive(prim);
            DNDS_check_throw_info(Rmix > 0, "primTPToConservative(): mixture gas constant must be positive");
            prim[0] = p_code / (Rmix * T_code);
            primToConservative(prim, cons, options);
        }

        /**
         * @brief Convert prescribed total pressure/temperature to static primitive state.
         *
         * The input/output primitive vector uses [rho, u, v, (w), p, Y_k].  The
         * velocity and composition entries are inputs; rho and p are overwritten.
         * If the requested velocity would consume more than 95% of total enthalpy,
         * the velocity magnitude is reduced, matching the historical BC safeguard.
         *
         * Non-reactive constant-gamma gas uses the closed-form isentropic formula.
         * Reactive ideal-gas mixtures iterate Cp, R, and gamma from the static
         * state at fixed inflow composition.
         */
        void totalToStaticPrimitive(real pTotal, real TTotal, TU &primStatic,
                                    const StateConversionOptions &options = StateConversionOptions{}) const
        {
            DNDS_assert_info(pTotal > 0 && TTotal > 0,
                             fmt::format("totalToStaticPrimitive(): invalid total p/T [{:.3e}, {:.3e}]",
                                         pTotal, TTotal));

            real vSqrRequest = primStatic(Seq123).squaredNorm();
            auto applyVelocityLimit = [&](real vSqrUse)
            {
                if (vSqrRequest > vSqrUse && vSqrRequest > 0)
                    primStatic(Seq123) *= std::sqrt(vSqrUse / vSqrRequest);
            };

            if (!hasChemicalSource())
            {
                real Cp = toCode(igProp_->CpGas());
                real Rgas = toCode(igProp_->Rgas);
                real gamma = igProp_->gamma;
                real vSqrUse = std::min(vSqrRequest, TTotal * 2 * Cp * 0.95);
                real TStatic = TTotal - 0.5 * vSqrUse / Cp;
                real pStatic = pTotal * std::pow(TStatic / TTotal, gamma / (gamma - 1));
                primStatic(0) = pStatic / std::max(Rgas * TStatic, 1e-60);
                primStatic(I4) = pStatic;
                applyVelocityLimit(vSqrUse);
                return;
            }

            DNDS_assert_info(chem().isIdealGas(),
                             "totalToStaticPrimitive(): reactive total-condition conversion requires ideal-gas EOS");
            DNDS_check_throw_info(std::isfinite(options.totalToStaticTolerance) && options.totalToStaticTolerance > 0,
                                  fmt::format("totalToStaticPrimitive(): invalid tolerance {:.3e}",
                                              options.totalToStaticTolerance));

            auto &c = chem();
            int Ns1 = c.nSpecies() - 1;
            int nVars = static_cast<int>(primStatic.size());
            int Isp = nVars - Ns1;
            std::vector<double> Ybuf(c.nSpecies());
            Chemistry::SpeciesBufferView Y{Ybuf.data(), c.nSpecies()};
            c.massFractions(1.0, primStatic.data() + Isp, Ns1, Y);

            double TTotalPhys = toPhysT(TTotal);
            double pTotalPhys = toPhysP(pTotal);
            double hTotal = c.mixtureEnthalpy(TTotalPhys, Y, pTotalPhys);
            double sTotal = c.mixtureEntropy(TTotalPhys, Y, pTotalPhys);
            double TminPhys = std::min(TTotalPhys * 0.999, std::max(c.minTemperature(), 1.0));
            double hMin = c.mixtureEnthalpy(TminPhys, Y, pTotalPhys);
            double maxKinetic = std::max(0.0, 0.95 * (hTotal - hMin));
            real vSqrUse = std::min(vSqrRequest, real(2.0 * maxKinetic / (igProp_->U0 * igProp_->U0)));
            double hTarget = hTotal - 0.5 * vSqrUse * igProp_->U0 * igProp_->U0;

            double TLo = TminPhys;
            double THi = TTotalPhys;
            double TStaticPhys = THi;
            int maxIterations = std::max(options.totalToStaticMaxIterations, 1);
            double finalErr = std::numeric_limits<double>::infinity();
            bool converged = false;
            for (int iter = 0; iter < maxIterations; ++iter)
            {
                TStaticPhys = 0.5 * (TLo + THi);
                double hMid = c.mixtureEnthalpy(TStaticPhys, Y, pTotalPhys);
                finalErr = std::abs(hMid - hTarget) / std::max(std::abs(hTarget), 1.0);
                if (finalErr < options.totalToStaticTolerance)
                {
                    converged = true;
                    break;
                }
                if (hMid < hTarget)
                    TLo = TStaticPhys;
                else
                    THi = TStaticPhys;
            }
            DNDS_check_throw_info(converged,
                                  fmt::format("totalToStaticPrimitive(): failed to converge after {} bisections; residual={:.3e}, tolerance={:.3e}, Ttotal={:.6e} K, ptotal={:.6e} Pa",
                                              maxIterations, finalErr, options.totalToStaticTolerance,
                                              TTotalPhys, pTotalPhys));

            real Rgas = toCode(c.mixtureR(Y));
            double sAtPtotal = c.mixtureEntropy(TStaticPhys, Y, pTotalPhys);
            double pStaticPhys = pTotalPhys * std::exp((sAtPtotal - sTotal) / std::max(c.mixtureR(Y), 1e-30));
            real TStatic = toCodeT(TStaticPhys);
            real pStatic = pStaticPhys / p0();
            primStatic(0) = pStatic / std::max(Rgas * TStatic, 1e-60);
            primStatic(I4) = pStatic;
            applyVelocityLimit(vSqrUse);
        }

        std::tuple<real, real> primitiveStaticToTotalPT(const TU &primStatic,
                                                        const StateConversionOptions &options = StateConversionOptions{}) const
        {
            real pStatic = primStatic(I4);
            real vSqr = primStatic(Seq123).squaredNorm();
            DNDS_assert_info(primStatic(0) > 0 && pStatic > 0,
                             fmt::format("primitiveStaticToTotalPT(): invalid rho/p [{:.3e}, {:.3e}]",
                                         primStatic(0), pStatic));

            if (!hasChemicalSource())
            {
                real gammaUse = igProp_->gamma;
                real Rgas = toCode(igProp_->Rgas);
                real TStatic = pStatic / std::max(primStatic(0) * Rgas, 1e-60);
                real asqr = gammaUse * pStatic / primStatic(0);
                real Msqr = vSqr / std::max(asqr, 1e-60);
                real factor = 1 + (gammaUse - 1) * 0.5 * Msqr;
                return {std::pow(factor, gammaUse / (gammaUse - 1)) * pStatic,
                        factor * TStatic};
            }

            DNDS_assert_info(chem().isIdealGas(),
                             "primitiveStaticToTotalPT(): reactive total-condition conversion requires ideal-gas EOS");
            DNDS_check_throw_info(std::isfinite(options.totalToStaticTolerance) && options.totalToStaticTolerance > 0,
                                  fmt::format("primitiveStaticToTotalPT(): invalid tolerance {:.3e}",
                                              options.totalToStaticTolerance));

            auto &c = chem();
            int Ns1 = c.nSpecies() - 1;
            int nVars = static_cast<int>(primStatic.size());
            int Isp = nVars - Ns1;
            std::vector<double> Ybuf(c.nSpecies());
            Chemistry::SpeciesBufferView Y{Ybuf.data(), c.nSpecies()};
            c.massFractions(1.0, primStatic.data() + Isp, Ns1, Y);

            double TStaticPhys = toPhysT(pStatic / std::max(primStatic(0) * toCode(c.mixtureR(Y)), 1e-60));
            double pStaticPhys = toPhysP(pStatic);
            double hTarget = c.mixtureEnthalpy(TStaticPhys, Y, pStaticPhys) + 0.5 * vSqr * igProp_->U0 * igProp_->U0;
            double sStatic = c.mixtureEntropy(TStaticPhys, Y, pStaticPhys);
            double TLo = std::max(c.minTemperature(), 1.0);
            double THi = std::max(TStaticPhys * 1.01, TStaticPhys + 1.0);
            while (c.mixtureEnthalpy(THi, Y, pStaticPhys) < hTarget)
                THi *= 1.5;

            double TTotalPhys = THi;
            int maxIterations = std::max(options.totalToStaticMaxIterations, 1);
            double finalErr = std::numeric_limits<double>::infinity();
            bool converged = false;
            for (int iter = 0; iter < maxIterations; ++iter)
            {
                TTotalPhys = 0.5 * (TLo + THi);
                double hMid = c.mixtureEnthalpy(TTotalPhys, Y, pStaticPhys);
                finalErr = std::abs(hMid - hTarget) / std::max(std::abs(hTarget), 1.0);
                if (finalErr < options.totalToStaticTolerance)
                {
                    converged = true;
                    break;
                }
                if (hMid < hTarget)
                    TLo = TTotalPhys;
                else
                    THi = TTotalPhys;
            }
            DNDS_check_throw_info(converged,
                                  fmt::format("primitiveStaticToTotalPT(): failed to converge after {} bisections; residual={:.3e}, tolerance={:.3e}",
                                              maxIterations, finalErr, options.totalToStaticTolerance));

            double Rmix = c.mixtureR(Y);
            double sAtStaticP = c.mixtureEntropy(TTotalPhys, Y, pStaticPhys);
            double pTotalPhys = pStaticPhys * std::exp((sAtStaticP - sStatic) / std::max(Rmix, 1e-30));
            return {pTotalPhys / p0(), toCodeT(TTotalPhys)};
        }

        /**
         * @brief Conservative → prim-TP.  Output: [T, u, v, (w), p, Y_k].
         * @note  For I/O purposes — not for performance-critical tight loops.
         */
        void conservativeToPrimTP(const TU &cons, TU &primTP,
                                  const StateConversionOptions &options = StateConversionOptions{}) const
        {
            if (hasChemicalSource())
                DNDS_assert_info(chem().isIdealGas(),
                                 "conservativeToPrimTP(): requires ideal-gas EOS");

            real T = temperature(cons, 0, options.temperatureUVTolerance);
            real gammaEqUse = gammaEq(T, cons);
            real rhoE_base = mixtureBaseInternalRhoE(cons);
            Gas::IdealGasThermalConservative2Primitive<dim>(cons, primTP, gammaEqUse, rhoE_base);
            primTP[0] = T;
        }

        // ---- Unit-scaling helpers (I/O only) ---------------------------------
        // Conservative vector layout: [rho, rhoU, rhoV, (rhoW), rhoE, RANS..., rhoY_0...]
        //   rho:       * rho0
        //   rhoU_j:    * rho0 * U0
        //   rhoE:      * rho0 * U0²
        //   RANS:      per-variable (see below)
        //   rhoY_k:    * rho0   (species conservative, trailing block)
        // Primitive helpers: species mass fractions Y_k are dimensionless (mass ratio),
        // so they are left unscaled in all primCodeToPhys/PrimToPhys helpers.
        // RANS conservative variable scaling (code↔phys):
        //   rho_nuTilde: * rho0            (nuTilde non-dimensional in code)
        //   rho_k:       * rho0 * U0²      (k: energy-like, [m²/s²])
        //   rho_omega:   * rho0 * U0 / L0  (omega: 1/t0 = U0/L0, [1/s])
        //   rho_epsilon: * rho0 * U0³ / L0 (epsilon: U0³/L0, [m²/s³])

        template <typename TVal>
        void consCodeToPhys(const TVal &code, TVal &phys) const
        {
            phys = code;
            phys[0] *= igProp_->rho0;
            for (int j = 1; j <= dim; ++j)
                phys[j] *= (igProp_->rho0 * igProp_->U0);
            phys[dim + 1] *= (igProp_->rho0 * igProp_->U0 * igProp_->U0);
            this->scaleRansConsCodeToPhys(phys);
            int Ns1 = hasChemicalSource() ? chem().nSpecies() - 1 : 0;
            int Isp = static_cast<int>(phys.size()) - Ns1;
            for (int k = Isp; k < (int)phys.size(); ++k)
                phys[k] *= igProp_->rho0;
        }

        template <typename TVal>
        void consPhysToCode(const TVal &phys, TVal &code) const
        {
            code = phys;
            code[0] /= igProp_->rho0;
            for (int j = 1; j <= dim; ++j)
                code[j] /= (igProp_->rho0 * igProp_->U0);
            code[dim + 1] /= (igProp_->rho0 * igProp_->U0 * igProp_->U0);
            this->scaleRansConsPhysToCode(code);
            int Ns1 = hasChemicalSource() ? chem().nSpecies() - 1 : 0;
            int Isp = static_cast<int>(code.size()) - Ns1;
            for (int k = Isp; k < (int)code.size(); ++k)
                code[k] /= igProp_->rho0;
        }

        // --- RANS model query (static trait first, then runtime set by setRANS) -
        RANSModel ransModel() const
        {
            if constexpr (EulerModelTraits<model>::hasSA)
                return RANS_SA;
            if constexpr (EulerModelTraits<model>::has2EQ)
            {
                if (ransModel_ == RANS_KOWilcox || ransModel_ == RANS_KOSST || ransModel_ == RANS_RKE)
                    return ransModel_;
                return RANS_KOWilcox; // default 2EQ
            }
            // dynamic (NS_EX): use ransModel_ (defaults to RANS_None)
            DNDS_assert(ransModel_ != RANS_Unknown);
            return ransModel_;
        }

        // --- RANS scaling provider (switches on RANS model) --------------------
        // Primitive / conservative scaling factors per variable position.
        //   pos = 0: always first RANS variable (nuTilde or k)
        //   pos = 1: second RANS variable (omega or epsilon), if 2EQ
        real ransPrimScaleCodeToPhys(int pos) const
        {
            switch (ransModel())
            {
            case RANS_SA:
                return real(1.0); // nuTilde: non-dimensional
            case RANS_KOWilcox:
            case RANS_KOSST:
                if (pos == 0)
                    return igProp_->U0 * igProp_->U0; // k: U0²
                if (pos == 1)
                    return igProp_->U0 / std::max(igProp_->L0, real(1e-60)); // omega: U0/L0 = 1/t0
                break;
            case RANS_RKE:
                if (pos == 0)
                    return igProp_->U0 * igProp_->U0; // k: U0²
                if (pos == 1)
                    return igProp_->U0 * igProp_->U0 * igProp_->U0 / std::max(igProp_->L0, real(1e-60)); // epsilon: U0³/L0
                break;
            default:
                break;
            }
            return real(1.0);
        }
        real ransPrimScalePhysToCode(int pos) const { return real(1.0) / std::max(ransPrimScaleCodeToPhys(pos), real(1e-60)); }
        real ransConsScaleCodeToPhys(int pos) const { return igProp_->rho0 * ransPrimScaleCodeToPhys(pos); }
        real ransConsScalePhysToCode(int pos) const { return real(1.0) / std::max(ransConsScaleCodeToPhys(pos), real(1e-60)); }

        template <typename TVal>
        void scaleRansConsCodeToPhys(TVal &vec) const
        {
            int n = nRANSVars();
            for (int j = 0; j < n; ++j)
                vec[dim + 2 + j] *= ransConsScaleCodeToPhys(j);
        }
        template <typename TVal>
        void scaleRansConsPhysToCode(TVal &vec) const
        {
            int n = nRANSVars();
            for (int j = 0; j < n; ++j)
                vec[dim + 2 + j] *= ransConsScalePhysToCode(j);
        }
        template <typename TVal>
        void scaleRansPrimCodeToPhys(TVal &vec) const
        {
            int n = nRANSVars();
            for (int j = 0; j < n; ++j)
                vec[dim + 2 + j] *= ransPrimScaleCodeToPhys(j);
        }
        template <typename TVal>
        void scaleRansPrimPhysToCode(TVal &vec) const
        {
            int n = nRANSVars();
            for (int j = 0; j < n; ++j)
                vec[dim + 2 + j] *= ransPrimScalePhysToCode(j);
        }

        template <typename TVal>
        void primCodeToPhys(const TVal &code, TVal &phys) const
        {
            phys = code;
            phys[0] *= igProp_->rho0;
            for (int j = 1; j <= dim; ++j)
                phys[j] *= igProp_->U0;
            phys[dim + 1] *= (igProp_->rho0 * igProp_->U0 * igProp_->U0);
            this->scaleRansPrimCodeToPhys(phys);
        }

        template <typename TVal>
        void primPhysToCode(const TVal &phys, TVal &code) const
        {
            code = phys;
            code[0] /= igProp_->rho0;
            for (int j = 1; j <= dim; ++j)
                code[j] /= igProp_->U0;
            code[dim + 1] /= (igProp_->rho0 * igProp_->U0 * igProp_->U0);
            this->scaleRansPrimPhysToCode(code);
        }

        template <typename TVal>
        void primRhoTCodeToPhys(const TVal &code, TVal &phys) const
        {
            phys = code;
            phys[0] *= igProp_->rho0;
            for (int j = 1; j <= dim; ++j)
                phys[j] *= igProp_->U0;
            phys[dim + 1] = toPhysT(code[dim + 1]);
            this->scaleRansPrimCodeToPhys(phys);
        }

        template <typename TVal>
        void primRhoTPhysToCode(const TVal &phys, TVal &code) const
        {
            code = phys;
            code[0] /= igProp_->rho0;
            for (int j = 1; j <= dim; ++j)
                code[j] /= igProp_->U0;
            code[dim + 1] = toCodeT(phys[dim + 1]);
            this->scaleRansPrimPhysToCode(code);
        }

        template <typename TVal>
        void primTPCodeToPhys(const TVal &code, TVal &phys) const
        {
            phys = code;
            phys[0] = toPhysT(code[0]);
            for (int j = 1; j <= dim; ++j)
                phys[j] *= igProp_->U0;
            phys[dim + 1] *= (igProp_->rho0 * igProp_->U0 * igProp_->U0);
            this->scaleRansPrimCodeToPhys(phys);
        }

        template <typename TVal>
        void primTPPhysToCode(const TVal &phys, TVal &code) const
        {
            code = phys;
            code[0] = toCodeT(phys[0]);
            for (int j = 1; j <= dim; ++j)
                code[j] /= igProp_->U0;
            code[dim + 1] /= (igProp_->rho0 * igProp_->U0 * igProp_->U0);
            this->scaleRansPrimPhysToCode(code);
        }

        void conservativeToStateValueOrigin(const TU &cons, TU &state, StateValueOrigin origin) const
        {
            TU tmp(cons.size());
            switch (origin)
            {
            case StateValueOrigin::Cons:
                state = cons;
                break;
            case StateValueOrigin::ConsSensible:
                consTotalToSensible(cons, state);
                break;
            case StateValueOrigin::PrimRhoP:
                conservativeToPrimitive(cons, state);
                break;
            case StateValueOrigin::PrimRhoT:
                conservativeToPrimRhoT(cons, state);
                break;
            case StateValueOrigin::PrimTP:
                conservativeToPrimTP(cons, state);
                break;
            case StateValueOrigin::ConsPhy:
                consCodeToPhys(cons, state);
                break;
            case StateValueOrigin::ConsSensiblePhy:
                consTotalToSensible(cons, tmp);
                consCodeToPhys(tmp, state);
                break;
            case StateValueOrigin::PrimRhoPPhy:
                conservativeToPrimitive(cons, tmp);
                primCodeToPhys(tmp, state);
                break;
            case StateValueOrigin::PrimRhoTPhy:
                conservativeToPrimRhoT(cons, tmp);
                primRhoTCodeToPhys(tmp, state);
                break;
            case StateValueOrigin::PrimTPPhy:
                conservativeToPrimTP(cons, tmp);
                primTPCodeToPhys(tmp, state);
                break;
            default:
                DNDS_check_throw_info(false, fmt::format("unsupported StateValueOrigin [{}]", StateValueOriginName(origin)));
            }
        }

        void stateValueOriginToConservative(const TU &state, TU &cons, StateValueOrigin origin) const
        {
            TU tmp(state.size());
            switch (origin)
            {
            case StateValueOrigin::Cons:
                cons = state;
                break;
            case StateValueOrigin::ConsSensible:
                consSensibleToTotal(state, cons);
                break;
            case StateValueOrigin::PrimRhoP:
                primToConservative(state, cons);
                break;
            case StateValueOrigin::PrimRhoT:
                primRhoTToConservative(state, cons);
                break;
            case StateValueOrigin::PrimTP:
                primTPToConservative(state, cons);
                break;
            case StateValueOrigin::ConsPhy:
                consPhysToCode(state, cons);
                break;
            case StateValueOrigin::ConsSensiblePhy:
                consPhysToCode(state, tmp);
                consSensibleToTotal(tmp, cons);
                break;
            case StateValueOrigin::PrimRhoPPhy:
                primPhysToCode(state, tmp);
                primToConservative(tmp, cons);
                break;
            case StateValueOrigin::PrimRhoTPhy:
                primRhoTPhysToCode(state, tmp);
                primRhoTToConservative(tmp, cons);
                break;
            case StateValueOrigin::PrimTPPhy:
                primTPPhysToCode(state, tmp);
                primTPToConservative(tmp, cons);
                break;
            default:
                DNDS_check_throw_info(false, fmt::format("unsupported StateValueOrigin [{}]", StateValueOriginName(origin)));
            }
        }

    private:
        void validatePrimitiveRhoP(const TU &prim, const char *label) const
        {
            static const int I4 = dim + 1;
            DNDS_check_throw_info(prim[0] > 0 && prim[I4] > 0,
                                  fmt::format("{}: rho and p must be positive", label));
        }

        TU sanitizePrimitiveSpecies(const TU &prim) const
        {
            TU ret = prim;
            if (!hasChemicalSource())
                return ret;
            auto &c = chem();
            int Ns1 = c.nSpecies() - 1;
            int nVars = static_cast<int>(ret.size());
            int Isp = nVars - Ns1;
            DNDS_check_throw_info(Isp >= dim + 2,
                                  "sanitizePrimitiveSpecies(): state vector too small for mechanism species count");
            std::vector<double> Ybuf(static_cast<size_t>(c.nSpecies()));
            Chemistry::SpeciesBufferView Y{Ybuf.data(), c.nSpecies()};
            c.massFractions(1.0, ret.data() + Isp, Ns1, Y);
            for (int k = 0; k < Ns1; ++k)
                ret[Isp + k] = Y[k];
            return ret;
        }

        /// Compute rhoE_base from a primitive vector [rho, u, v, (w), p/T, Y_k].
        real baseInternalFromPrimitive(const TU &prim) const
        {
            if (!hasChemicalSource())
                return 0;
            auto &c = chem();
            int Ns1 = c.nSpecies() - 1;
            int nVars = static_cast<int>(prim.size());
            int Isp = nVars - Ns1;
            std::vector<double> Ybuf(static_cast<size_t>(c.nSpecies()));
            Chemistry::SpeciesBufferView Y{Ybuf.data(), c.nSpecies()};
            c.massFractions(1.0, prim.data() + Isp, Ns1, Y);
            return c.mixtureBaseInternalRhoE(prim[0], Y);
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
            std::vector<double> Ybuf(static_cast<size_t>(c.nSpecies()));
            Chemistry::SpeciesBufferView Y{Ybuf.data(), c.nSpecies()};
            c.massFractions(1.0, prim.data() + Isp, Ns1, Y); // rho=1 → Y_k directly
            return toCode(c.mixtureR(Y));
        }

    public:
        // ---- Transport (mixture) --------------------------------------------

        /// Sutherland + const + density-proportional fallback, or Cantera mixture-averaged.
        real mixtureViscosity(real T, real p, const TU &U) const;

        real mixtureConductivity(real T, real p, const TU &U) const;

        template <class TD>
        void mixtureDiffusivity(real T, real p, const TU &U, TD &&D) const;

        real speciesDiffusivityK(real T, real p, const TU &U, int k) const;

        const std::string &transportModel() const
        {
            DNDS_assert(hasChemicalSource());
            return chem().transportModel();
        }

        bool isMixtureAveragedTransport() const
        {
            return hasChemicalSource() && chem().isMixtureAveragedTransport();
        }

        struct MixtureAveragedDiffusionBuffers
        {
            std::vector<real> h;
            std::vector<real> D;
            std::vector<real> Y;
            std::vector<real> JRawN;

            void resize(int nSpecies)
            {
                h.resize(static_cast<size_t>(nSpecies));
                D.resize(static_cast<size_t>(nSpecies));
                Y.resize(static_cast<size_t>(nSpecies));
                JRawN.resize(static_cast<size_t>(nSpecies));
            }
        };

        /** Add mixture-averaged species diffusion and enthalpy transport to a viscous flux slot.
         *  The solver convention is F_total = F_inviscid - VisFlux, so this stores -J_k in
         *  species equations and -Σ h_k J_k in the energy equation. Also corrects the
         *  conductive heat flux for ∇R(Y): k∇T includes -k*T/R*∇R.
         */
        template <class TGradUPrim, class TNorm, class TFlux>
        void addMixtureAveragedSpeciesDiffusionFlux(real T, real p, const TU &U,
                                                    const TGradUPrim &GradUPrim,
                                                    const TNorm &norm,
                                                    real thermalConductivity,
                                                    bool adiabaticWall,
                                                    bool impermeableWall,
                                                    MixtureAveragedDiffusionBuffers &buffers,
                                                    TFlux &visFlux) const
        {
            if (!hasChemicalSource())
                return;
            DNDS_check_throw_info(isMixtureAveragedTransport(),
                                  fmt::format("Reactive species diffusion currently implements only mixture-averaged transport; requested [{}]",
                                              transportModel()));

            static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
            static const auto I4 = dim + 1;

            auto &c = chem();
            int Ns = c.nSpecies();
            int Ns1 = Ns - 1;
            int nV = static_cast<int>(U.size());
            int Isp = nV - Ns1;
            buffers.resize(Ns);

            // Energy diffusion uses full species enthalpy h_k(T,p,Y), not the
            // bookkeeping base-energy/rhoE_base offset. This is the old rhoH
            // diffusion concern: the correct term is -Σ h_k J_k in physical flux.
            Chemistry::SpeciesBufferView hv{buffers.h.data(), Ns};
            speciesEnthalpies(T, p, U, hv);

            mixtureDiffusivity(T, p, U, buffers.D);

            real rhoFace = U(0);
            real rhoInvFace = 1.0 / std::max(rhoFace, verySmallReal);
            std::fill(buffers.Y.begin(), buffers.Y.end(), real(0));
            std::fill(buffers.JRawN.begin(), buffers.JRawN.end(), real(0));
            real sumY = 0.0;
            real sumGradYDotN = 0.0;
            real sumJRawN = 0.0;
            real gradRDotN = 0.0;
            real RLast = speciesGasConstantK(Ns - 1);

            for (int kk = 0; kk < Ns1; ++kk)
            {
                buffers.Y[kk] = U(Isp + kk) * rhoInvFace;
                sumY += buffers.Y[kk];
                real gradYkDotN = GradUPrim(Seq012, Isp + kk).dot(norm);
                gradRDotN += (speciesGasConstantK(kk) - RLast) * gradYkDotN;
                buffers.JRawN[kk] = -rhoFace * buffers.D[kk] * gradYkDotN;
                sumGradYDotN += gradYkDotN;
                sumJRawN += buffers.JRawN[kk];
            }

            // Conductive heat flux sign: VisFlux stores +k∇T·n and total flux subtracts VisFlux.
            // Since T = p/(rho R), the mixture correction is -k*T/R*∇R·n.
            if (!adiabaticWall)
                visFlux(I4) -= thermalConductivity * T / std::max(Rgas(U), verySmallReal) * gradRDotN;

            buffers.Y[Ns - 1] = real(1) - sumY;
            buffers.JRawN[Ns - 1] = rhoFace * buffers.D[Ns - 1] * sumGradYDotN;
            sumJRawN += buffers.JRawN[Ns - 1];

            real vcN = -sumJRawN * rhoInvFace;
            if (!impermeableWall)
            {
                for (int kk = 0; kk < Ns; ++kk)
                {
                    real JkN = buffers.JRawN[kk] + rhoFace * buffers.Y[kk] * vcN;
                    real FvSpeciesN = -JkN;
                    if (kk < Ns1)
                        visFlux(Isp + kk) += FvSpeciesN;
                    visFlux(I4) += buffers.h[kk] * FvSpeciesN;
                }
            }
        }

        // ---- Kinetics accessor ----------------------------------------------

        /// number of species; 1 for non-extended
        int nSpecies() const { return hasChemicalSource() ? chem().nSpecies() : 1; }
        const std::string &speciesName(int k) const
        {
            DNDS_assert(hasChemicalSource());
            return chem().speciesNames()[static_cast<size_t>(k)];
        }
        real speciesGasConstantK(int k) const
        {
            DNDS_assert(hasChemicalSource());
            return toCode(chem().speciesGasConstants()[static_cast<size_t>(k)]);
        }

        void advanceAffineConstVolumeY(real &T, real rho,
                                       Chemistry::SpeciesBufferView Y,
                                       real chemistryScale,
                                       real linearTime,
                                       Chemistry::ConstSpeciesBufferView constantTerm,
                                       real advanceTime,
                                       real rtol = 1e-10,
                                       real atol = 1e-18,
                                       int maxOrder = 1,
                                       int maxSteps = 2000) const
        {
            DNDS_assert(hasChemicalSource());
            std::vector<double> cPhys(static_cast<size_t>(constantTerm.nSpecies));
            for (int k = 0; k < constantTerm.nSpecies; ++k)
                cPhys[static_cast<size_t>(k)] = constantTerm[k] / t0();
            double TPhys = toPhysT(T);
            chem().advanceAffineConstVolume(
                TPhys, rho * igProp_->rho0, Y, chemistryScale,
                linearTime * t0(),
                Chemistry::ConstSpeciesBufferView{cPhys.data(), static_cast<int>(cPhys.size())},
                advanceTime * t0(), rtol, atol, maxOrder, maxSteps);
            T = toCodeT(TPhys);
        }

        void advanceConstVolumeY(real &T, real rho,
                                 Chemistry::SpeciesBufferView Y,
                                 real chemistryScale,
                                 real advanceTime,
                                 real rtol = 1e-10,
                                 real atol = 1e-18,
                                 int maxOrder = 0,
                                 int maxSteps = 2000) const
        {
            DNDS_assert(hasChemicalSource());
            double TPhys = toPhysT(T);
            chem().advanceConstVolume(
                TPhys, rho * igProp_->rho0, Y,
                chemistryScale, advanceTime * t0(), rtol, atol, maxOrder, maxSteps);
            T = toCodeT(TPhys);
        }

        /// Per-species total specific enthalpies in code units (h_k/U0²).
        /// For ideal-gas EOS: h_k = e_sensible_k + h_f_k + R_k·T  (no KE term).
        /// For non-ideal EOS, Cantera's speciesEnthalpies includes EOS-specific
        /// non-ideal contributions; the additive decomposition above does not hold.
        /// Sum_k Y_k·h_k = H_mixture.
        void speciesEnthalpies(real T, real p, const TU &U, Chemistry::SpeciesBufferView h) const
        {
            DNDS_assert(hasChemicalSource());
            auto &c = chem();
            int Ns1 = c.nSpecies() - 1;
            int nVars = static_cast<int>(U.size());
            int Isp = nVars - Ns1;
            std::vector<double> Ybuf(static_cast<size_t>(c.nSpecies()));
            Chemistry::SpeciesBufferView Y{Ybuf.data(), c.nSpecies()};
            c.massFractions(U[0], U.data() + Isp, Ns1, Y);
            c.speciesEnthalpies(toPhysT(T), toPhysP(p), Y, h);
            real invU0sq = 1.0 / (igProp_->U0 * igProp_->U0);
            for (int k = 0; k < c.nSpecies(); ++k)
                h[k] *= invU0sq;
        }

        /// Per-species base internal energies in code units (e_base,k / U0²) as Eigen Map.
        /// Sum equals mixtureBaseInternalRhoE(U)/rho. Returns empty Map if no chemistry.
        Eigen::Map<const Eigen::Vector<real, Eigen::Dynamic>> mixtureBaseInternalRhoESpecies() const
        {
            if (!hasChemicalSource())
                return {nullptr, 0};
            auto &c = chem();
            auto v = c.mixtureBaseInternalRhoESpecies();
            return Eigen::Map<const Eigen::Vector<real, Eigen::Dynamic>>(v.data, v.nSpecies);
        }

    private:
        std::vector<double> massFractionsVector(const TU &U) const
        {
            auto &c = chem();
            int Ns1 = c.nSpecies() - 1;
            int nVars = static_cast<int>(U.size());
            int Isp = nVars - Ns1;
            std::vector<double> Y(static_cast<size_t>(c.nSpecies()));
            c.massFractions(U[0], U.data() + Isp, Ns1, {Y.data(), c.nSpecies()});
            return Y;
        }

        ChemPtr pool_;
        std::unique_ptr<const IdealGas> igProp_;
        int nRANS_ = 0;                   ///< runtime RANS count (0=unset or none, 1=SA, 2=2EQ)
        RANSModel ransModel_ = RANS_None; ///< runtime RANS model selection
    };

    // ========================================================================
    // Out-of-line template implementations
    // ========================================================================

    template <EulerModel model>
    void PhysicsProperties<model>::resolveStateValue(StateValue &value, int nVars,
                                                     std::ostream *os,
                                                     const std::string &label) const
    {
        static const int I4 = dim + 1;
        value.checkSize(nVars, label);
        value.keepOnlyOrigin();
        DNDS_check_throw_info(value.originType != StateValueOrigin::None &&
                                  value.originType != StateValueOrigin::NonState &&
                                  value.originType != StateValueOrigin::Invalid,
                              fmt::format("{} has unsupported state type [{}]", label, StateValueOriginName(value.originType)));
        DNDS_check_throw_info(StateValue::filled(value.originVector()),
                              fmt::format("{} has an empty or non-finite state value", label));
        value.fillMissingWithNaN(nVars);

        auto consPhysToCode = [&](const Eigen::Vector<real, -1> &v)
        {
            Eigen::Vector<real, -1> o;
            this->consPhysToCode(v, o);
            return o;
        };
        auto consCodeToPhys = [&](const Eigen::Vector<real, -1> &v)
        {
            Eigen::Vector<real, -1> o;
            this->consCodeToPhys(v, o);
            return o;
        };
        auto primRhoPPhysToCode = [&](const Eigen::Vector<real, -1> &v)
        {
            Eigen::Vector<real, -1> o;
            this->primPhysToCode(v, o);
            return o;
        };
        auto primRhoPCodeToPhys = [&](const Eigen::Vector<real, -1> &v)
        {
            Eigen::Vector<real, -1> o;
            this->primCodeToPhys(v, o);
            return o;
        };
        auto primRhoTPhysToCode = [&](const Eigen::Vector<real, -1> &v)
        {
            Eigen::Vector<real, -1> o;
            this->primRhoTPhysToCode(v, o);
            return o;
        };
        auto primRhoTCodeToPhys = [&](const Eigen::Vector<real, -1> &v)
        {
            Eigen::Vector<real, -1> o;
            this->primRhoTCodeToPhys(v, o);
            return o;
        };
        auto primTPPhysToCode = [&](const Eigen::Vector<real, -1> &v)
        {
            Eigen::Vector<real, -1> o;
            this->primTPPhysToCode(v, o);
            return o;
        };
        auto primTPCodeToPhys = [&](const Eigen::Vector<real, -1> &v)
        {
            Eigen::Vector<real, -1> o;
            this->primTPCodeToPhys(v, o);
            return o;
        };

        auto toTU = [&](const Eigen::Vector<real, -1> &v)
        {
            TU out(v.size());
            out = v;
            return out;
        };
        auto fromTU = [](const TU &v)
        {
            Eigen::Vector<real, -1> out(v.size());
            out = v;
            return out;
        };

        switch (value.originType)
        {
        case StateValueOrigin::Cons:
            break;
        case StateValueOrigin::ConsSensible:
        {
            TU in = toTU(value.consSensible), out(nVars);
            consSensibleToTotal(in, out);
            value.cons = fromTU(out);
            break;
        }
        case StateValueOrigin::PrimRhoP:
        {
            TU in = toTU(value.primRhoP), out(nVars);
            primToConservative(in, out);
            value.cons = fromTU(out);
            break;
        }
        case StateValueOrigin::PrimRhoT:
        {
            TU in = toTU(value.primRhoT), out(nVars);
            primRhoTToConservative(in, out);
            value.cons = fromTU(out);
            break;
        }
        case StateValueOrigin::PrimTP:
        {
            TU in = toTU(value.primTP), out(nVars);
            primTPToConservative(in, out);
            value.cons = fromTU(out);
            break;
        }
        case StateValueOrigin::ConsPhy:
            value.cons = consPhysToCode(value.cons_phy);
            break;
        case StateValueOrigin::ConsSensiblePhy:
        {
            value.consSensible = consPhysToCode(value.consSensible_phy);
            TU in = toTU(value.consSensible), out(nVars);
            consSensibleToTotal(in, out);
            value.cons = fromTU(out);
            break;
        }
        case StateValueOrigin::PrimRhoPPhy:
        {
            value.primRhoP = primRhoPPhysToCode(value.primRhoP_phy);
            TU in = toTU(value.primRhoP), out(nVars);
            primToConservative(in, out);
            value.cons = fromTU(out);
            break;
        }
        case StateValueOrigin::PrimRhoTPhy:
        {
            value.primRhoT = primRhoTPhysToCode(value.primRhoT_phy);
            TU in = toTU(value.primRhoT), out(nVars);
            primRhoTToConservative(in, out);
            value.cons = fromTU(out);
            break;
        }
        case StateValueOrigin::PrimTPPhy:
        {
            value.primTP = primTPPhysToCode(value.primTP_phy);
            TU in = toTU(value.primTP), out(nVars);
            primTPToConservative(in, out);
            value.cons = fromTU(out);
            break;
        }
        default:
            DNDS_assert_info(false, "StateValue has no originType");
        }

        DNDS_assert_info(StateValue::filled(value.cons), label + " did not resolve to cons");
        {
            TU in = toTU(value.cons), out(nVars);
            consTotalToSensible(in, out);
            value.consSensible = fromTU(out);
            conservativeToPrimitive(in, out);
            value.primRhoP = fromTU(out);
            conservativeToPrimRhoT(in, out);
            value.primRhoT = fromTU(out);
            conservativeToPrimTP(in, out);
            value.primTP = fromTU(out);
        }
        value.cons_phy = consCodeToPhys(value.cons);
        value.consSensible_phy = consCodeToPhys(value.consSensible);
        value.primRhoP_phy = primRhoPCodeToPhys(value.primRhoP);
        value.primRhoT_phy = primRhoTCodeToPhys(value.primRhoT);
        value.primTP_phy = primTPCodeToPhys(value.primTP);

        if (os)
        {
            nlohmann::ordered_json j = value;
            *os << fmt::format("Resolved state [{}] origin [{}]:\n{}\n",
                               label, StateValueOriginName(value.originType), j.dump(2));
            *os << fmt::format("  uConsCode: {}\n", fmt::streamed(value.cons.transpose()));
            *os << fmt::format("  uConsPhys: {}\n", fmt::streamed(value.cons_phy.transpose()));
            *os << fmt::format("  uTPCode:   {}\n", fmt::streamed(value.primTP.transpose()));
            *os << fmt::format("  uTPPhys:   {}\n", fmt::streamed(value.primTP_phy.transpose()));
        }
    }

    template <EulerModel model>
    void PhysicsProperties<model>::printInfo(std::ostream &os) const
    {
        auto &ig = *igProp_;
        os << fmt::format("=== PhysicsProperties (IdealGas) Info ===\n");
        os << fmt::format("  gamma:        {:.6e}\n", ig.gamma);
        os << fmt::format("  Rgas:         {:.6e} J/(kg*K)\n", ig.Rgas);
        os << fmt::format("  muGas:        {:.6e} Pa*s\n", ig.muGas);
        os << fmt::format("  Pr_gas:       {:.6e}\n", ig.prGas);
        os << fmt::format("  TRef:         {:.6e} K\n", ig.TRef);
        os << fmt::format("  C_Sutherland: {:.6e}\n", ig.CSutherland);
        os << fmt::format("  muModel:      {:d}\n", ig.muModel);
        os << fmt::format("\n");
        os << fmt::format("  Reference scales:\n");
        os << fmt::format("    T0:         {:.6e} K\n", ig.T0);
        os << fmt::format("    rho0:       {:.6e} kg/m^3\n", ig.rho0);
        os << fmt::format("    U0:         {:.6e} m/s\n", ig.U0);
        os << fmt::format("    L0:         {:.6e} m\n", ig.L0);
        os << fmt::format("  Derived scales:\n");
        os << fmt::format("    p0  = rho0*U0^2:              {:.6e} Pa\n", p0());
        os << fmt::format("    t0  = L0/U0:                   {:.6e} s\n", ig.L0 / ig.U0);
        os << fmt::format("    R0  = U0^2/T0:                 {:.6e} J/(kg*K)\n", R0());
        os << fmt::format("    mu0 = rho0*U0*L0:              {:.6e} Pa*s\n", ig.rho0 * ig.U0 * ig.L0);
        os << fmt::format("  Code Rgas:   {:.6e}\n", ig.Rgas / R0());
        os << fmt::format("  Code muGas:  {:.6e}\n", ig.muGas / (ig.rho0 * ig.U0 * ig.L0));
        os << fmt::format("  RANS model:  {:d}\n", static_cast<int>(ransModel_));
        os << fmt::format("  nRANS vars:  {:d}\n", nRANS_);
        if (hasChemicalSource())
        {
            os << fmt::format("\n");
            os << fmt::format("  nSpecies:    {:d}\n", nSpecies());
            chem().printInfo(os);
        }
    }

    template <EulerModel model>
    real PhysicsProperties<model>::mixtureViscosity(real T, real p, const TU &U) const
    {
        if (!hasChemicalSource())
        {
            switch (igProp_->muModel)
            {
            case 0:
                return igProp_->muGas;
            case 1: // Sutherland: μ = μ_ref * (T/T_ref)^1.5 * (T_ref + C) / (T + C)
            {
                real TRefCode = toCodeT(igProp_->TRef);
                real CSuthCode = toCodeT(igProp_->CSutherland);
                real TRel = T / TRefCode;
                return igProp_->muGas * TRel * std::sqrt(TRel) * (TRefCode + CSuthCode) / (T + CSuthCode);
            }
            case 2:
                return igProp_->muGas * U[0];
            default:
                DNDS_assert_info(false, fmt::format("mixtureViscosity: unrecognized muModel={}", igProp_->muModel));
                return igProp_->muGas;
            }
        }
        auto Y = massFractionsVector(U);
        real muPhys = chem().viscosity(toPhysT(T), toPhysP(p), {Y.data(), static_cast<int>(Y.size())});
        return muPhys / mu0();
    }

    template <EulerModel model>
    real PhysicsProperties<model>::mixtureConductivity(real T, real p, const TU &U) const
    {
        if (!hasChemicalSource())
            return Cp(T, U) * mixtureViscosity(T, p, U) / Pr();
        auto Y = massFractionsVector(U);
        real kPhys = chem().thermalConductivity(toPhysT(T), toPhysP(p), {Y.data(), static_cast<int>(Y.size())});
        return kPhys / k0();
    }

    template <EulerModel model>
    template <class TD>
    void PhysicsProperties<model>::mixtureDiffusivity(real T, real p, const TU &U, TD &&D) const
    {
        if (!hasChemicalSource())
            return;
        DNDS_check_throw_info(isMixtureAveragedTransport(),
                              fmt::format("mixtureDiffusivity(): only mixture-averaged transport is implemented; requested [{}]",
                                          transportModel()));
        int Ns = chem().nSpecies();
        std::vector<real> Dbuf(Ns);
        Chemistry::SpeciesBufferView Dv{Dbuf.data(), Ns};
        auto Y = massFractionsVector(U);
        chem().speciesDiffusivity(toPhysT(T), toPhysP(p), {Y.data(), static_cast<int>(Y.size())}, Dv);
        for (int k = 0; k < Ns; ++k)
            D[k] = Dbuf[k] / D0();
    }

    template <EulerModel model>
    real PhysicsProperties<model>::speciesDiffusivityK(real T, real p, const TU &U, int k) const
    {
        if (!hasChemicalSource())
        {
            // Non-reactive: Schmidt-number=1 diffusivity from mixture viscosity.
            // muModel=0,1,2 all work through mixtureViscosity(); Cantera path unused.
            real mu = mixtureViscosity(T, p, U);
            return mu / std::max(real(U[0]), 1e-60); // Sc = 1
        }
        DNDS_check_throw_info(isMixtureAveragedTransport(),
                              fmt::format("speciesDiffusivityK(): only mixture-averaged transport is implemented; requested [{}]",
                                          transportModel()));
        int Ns = chem().nSpecies();
        std::vector<real> Dbuf(Ns);
        Chemistry::SpeciesBufferView Dv{Dbuf.data(), Ns};
        auto Y = massFractionsVector(U);
        chem().speciesDiffusivity(toPhysT(T), toPhysP(p), {Y.data(), static_cast<int>(Y.size())}, Dv);
        return Dbuf[k] / D0();
    }

    // ========================================================================
    // Per-point EOS coefficients — unified constant / mixture dispatch
    // ========================================================================

    template <EulerModel model>
    real PhysicsProperties<model>::temperature(const TU &U, real TGuess, real uvTolerance) const
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
            DNDS_assert_info(uInternal > 0,
                             fmt::format("temperature(): non-reactive uInternal={:.3e} ≤ 0; "
                                         "rho={:.3e} rhoE={:.3e} vel2={:.3e}",
                                         uInternal, rho, U[I4], vel2));
            real p = (igProp_->gamma - 1) * rho * uInternal;
            return p * rhoInv / toCode(igProp_->Rgas);
        }
        real uPhys = uInternal * igProp_->U0 * igProp_->U0;
        DNDS_assert_info(chem().isIdealGas(), "temperature(): non-ideal-gas EOS conversion not yet implemented");
        real vPhys = rhoInv / igProp_->rho0;
        double T_guess = TGuess > 0 ? toPhysT(TGuess) : 200.0;
        // Fallback T guess using constant-gamma p/(rho*R) estimate; only executed
        // when warm-start T_guess is invalid (≤0), not a hot path.
        if (T_guess <= 0)
        {
            real uSensible = sensibleRhoE(U, I4) * rhoInv - 0.5 * vel2;
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
        auto Y = massFractionsVector(U);
        double Tphys = chem().temperatureFromUV(uPhys, vPhys, {Y.data(), static_cast<int>(Y.size())}, T_guess, uvTolerance);
        return toCodeT(Tphys);
    }

    template <EulerModel model>
    real PhysicsProperties<model>::gamma(real T, const TU &U) const
    {
        if (!hasChemicalSource())
            return igProp_->gamma;
        // Ideal-gas pressure: p = ρ_phys · R_phys · T_phys.  Non-ideal EOS
        // paths would need the actual pressure from the primitive state.
        auto Y = massFractionsVector(U);
        Chemistry::ConstSpeciesBufferView Yv{Y.data(), static_cast<int>(Y.size())};
        double pPhys = U(0) * igProp_->rho0 * chem().mixtureR(Yv) * toPhysT(T);
        return chem().mixtureGamma(toPhysT(T), Yv, pPhys);
    }

    template <EulerModel model>
    real PhysicsProperties<model>::Rgas(const TU &U) const
    {
        if (!hasChemicalSource())
            return toCode(igProp_->Rgas);
        auto Y = massFractionsVector(U);
        return toCode(chem().mixtureR({Y.data(), static_cast<int>(Y.size())}));
    }

    template <EulerModel model>
    real PhysicsProperties<model>::Cp(real T, const TU &U) const
    {
        if (!hasChemicalSource())
            return toCode(igProp_->CpGas());
        auto Y = massFractionsVector(U);
        Chemistry::ConstSpeciesBufferView Yv{Y.data(), static_cast<int>(Y.size())};
        double pPhys = U(0) * igProp_->rho0 * chem().mixtureR(Yv) * toPhysT(T);
        return toCode(chem().mixtureCp(toPhysT(T), Yv, pPhys));
    }

    template <EulerModel model>
    real PhysicsProperties<model>::Cv(real T, const TU &U) const
    {
        if (!hasChemicalSource())
            return Cp(T, U) - Rgas(U);
        auto Y = massFractionsVector(U);
        Chemistry::ConstSpeciesBufferView Yv{Y.data(), static_cast<int>(Y.size())};
        double pPhys = U(0) * igProp_->rho0 * chem().mixtureR(Yv) * toPhysT(T);
        return toCode(chem().mixtureCv(toPhysT(T), Yv, pPhys));
    }

} // namespace DNDS::Euler
