/**
 * @file ChemicalSource.hpp
 * @brief PIMPL-wrapped Cantera chemical kinetics, mixture properties, and transport.
 *
 * Public header — zero Cantera includes. The implementation (.cpp) owns the
 * Cantera Solution and fills user-supplied buffers. Buffer views are plain
 * pointer+size structs for zero-overhead interop with Eigen::Map.
 */

#pragma once

#include <memory>
#include <vector>
#include <string>

namespace DNDS::Euler::Chemistry
{

    /**
     * @brief Thin read-only view over a species-indexed buffer (mass fractions,
     *        production rates, diffusivities, etc.). Caller owns the storage.
     */
    struct SpeciesBufferView
    {
        double *data = nullptr;
        int nSpecies = 0;

        double &operator[](int i) { return data[i]; }
        double operator[](int i) const { return data[i]; }
    };

    /// Const version for input arrays.
    struct ConstSpeciesBufferView
    {
        const double *data = nullptr;
        int nSpecies = 0;

        double operator[](int i) const { return data[i]; }
    };

    /**
     * @brief Dense Jacobian view: Ns rows × nCols cols, column-major.
     *        Caller owns the storage.
     */
    struct JacobianBufferView
    {
        double *data = nullptr;
        int rows = 0; // Ns
        int cols = 0; // nVars
        int ld = 0;   // leading dimension (== rows for dense ColMajor)

        double &operator()(int i, int j) { return data[i + j * ld]; }
        double operator()(int i, int j) const { return data[i + j * ld]; }
    };

    /**
     * @brief PIMPL wrapper around Cantera (thermo + kinetics + transport).
     *
     * Construction loads a YAML mechanism. All evaluation methods take
     * temperature, pressure, and species mass fractions as input and write
     * results into caller-owned buffers. No Cantera headers are exposed.
     */
    class ChemicalSource
    {
    public:
        ChemicalSource();
        explicit ChemicalSource(const std::string &mechanismFile,
                                const std::string &phaseName = "");
        ~ChemicalSource();

        // Non-copyable, movable
        ChemicalSource(const ChemicalSource &) = delete;
        ChemicalSource &operator=(const ChemicalSource &) = delete;
        ChemicalSource(ChemicalSource &&) noexcept;
        ChemicalSource &operator=(ChemicalSource &&) noexcept;

        /// Deep-clone for thread-safety: creates independent Cantera Solution objects
        /// from the same mechanism file (no shared state between instances).
        std::unique_ptr<ChemicalSource> clone() const;

        int nSpecies() const;
        int nReactions() const;
        const std::vector<std::string> &speciesNames() const;
        const std::vector<double> &molecularWeights() const;

        // ---- Mixture thermodynamic properties (perfect gas, variable γ) ----

        double mixtureR(ConstSpeciesBufferView Y) const;
        double mixtureCp(double T, ConstSpeciesBufferView Y) const;
        double mixtureCv(double T, ConstSpeciesBufferView Y) const;
        double mixtureGamma(double T, ConstSpeciesBufferView Y) const;
        double speedOfSound(double T, ConstSpeciesBufferView Y) const;

        /** Specific internal energy [J/kg] at (T, p=1 atm, Y). */
        double mixtureIntEnergy(double T, ConstSpeciesBufferView Y) const;

        /** Specific enthalpy [J/kg] at (T, p=1 atm, Y). */
        double mixtureEnthalpy(double T, ConstSpeciesBufferView Y) const;

        /**
         * Solve T from specific internal energy u [J/kg] and specific volume v [m³/kg].
         * Uses Cantera setState_UV (Newton). Optional T_guess [K] as warm-start.
         */
        double temperatureFromUV(double u, double v,
                                 ConstSpeciesBufferView Y,
                                 double T_guess = 0) const;

        // ---- Kinetics ----

        enum JacobianFlags : int
        {
            JAC_DEFAULT = 0,
            JAC_SKIP_FLUID = 1 << 0,      // zero out ρ, ρu_j, ρE columns
            JAC_SKIP_ABSORPTION = 1 << 1, // don't absorb last-species into independent rows;
                                          // N2 row is filled, absorption terms omitted.
        };

        /** Net production rates ω_i [kmol/m³/s]. omega must have nSpecies elements. */
        void productionRates(double T, double p,
                             ConstSpeciesBufferView Y,
                             SpeciesBufferView omega) const;

        /**
         * Production rates AND Jacobian ∂ω/∂U.
         *   U = [ρ, ρu, ρv, {ρw,} ρE, ρY_0..ρY_{Ns-2}]
         * dOmegadU: Ns × nVars, column-major.
         * iEnergy = index of ρE in U (dim+1); species start = iEnergy+1.
         * velScale = U0 (m/s); rhoE, rhoU/V/W are code-scaled.
         * rhoScale = rho0 (kg/m³); needed for dC_k/d(rhoY_k)_code = rho0/MW_k.
         * jacFlags = bitmask of JacobianFlags.
         */
        void productionRatesAndJacobian(double T, double p, double rho,
                                        double rhoE, double rhoU, double rhoV, double rhoW,
                                        int iEnergy, double velScale, double rhoScale,
                                        ConstSpeciesBufferView Y,
                                        SpeciesBufferView omega,
                                        JacobianBufferView dOmegadU,
                                        int jacFlags = 0) const;

        // ---- Transport ----

        double viscosity(double T, double p, ConstSpeciesBufferView Y) const;
        double thermalConductivity(double T, double p, ConstSpeciesBufferView Y) const;

        /** Mixture-averaged species diffusivities [m²/s]. D must have nSpecies elements. */
        void speciesDiffusivity(double T, double p,
                                ConstSpeciesBufferView Y,
                                SpeciesBufferView D) const;

        /** Per-species specific enthalpies [J/kg]. h must have nSpecies elements. */
        void speciesEnthalpies(double T, double p,
                               ConstSpeciesBufferView Y,
                               SpeciesBufferView h) const;

        /** Per-species formation enthalpies [J/kg] at 298 K (constant, pre-cached). */
        void speciesFormationEnthalpies(SpeciesBufferView hf) const;

        /** Mixture formation energy Σ Y_k * h_f_k [J/kg]. In physical (SI) units. */
        double mixtureFormationEnergy(ConstSpeciesBufferView Y) const;

        /** EOS-agnostic enthalpy-internal-energy difference at reference T=298.15 K, p=1 atm.
         *  Returns h_mix(298,Y) − u_mix(298,Y) for the given composition [J/kg].
         *  For ideal gas this equals R_mix·298.15; for real gases uses the full EOS. */
        double pVAtReference(ConstSpeciesBufferView Y) const;

        /**
         * Sensible internal energy of the mixture at reference T=298.15 K,
         * assuming ideal-gas energy convention (e_sensible = 0 at T = 0 K).
         * Computed as cv_mass(T_ref, Y) · 298.15 [J/kg] via Cantera's own EOS.
         * Exact for ideal-gas thermo phases; approximate otherwise.
         */
        double sensibleInternalEnergyAtReference(ConstSpeciesBufferView Y) const;

        /** Whether the Cantera thermo phase uses an ideal-gas EOS. */
        bool isIdealGas() const;

        // ---- Per-instance buffers (thread-safe when each thread has its own) ----

        /** Compute mass fractions from conservative species densities. Fills bufY_, returns view into it. */
        ConstSpeciesBufferView massFractions(double rho, const double *rhoYK, int nTransported) const;

        /** Populate and return bufHf_ with per-species formation enthalpies in code units (hf_k/U0²).
         *  Needs invU0sq = 1/U0² to convert from physical [J/kg] to code units. */
        ConstSpeciesBufferView mixtureFormationRhoESpecies(double invU0sq) const;

        /** Code-scaled volumetric formation enthalpy: rho · Σ Y_k · hf_k_code. */
        double mixtureFormationRhoE(double rho, ConstSpeciesBufferView Y) const;

        /** Linearized increment of code-scaled formation enthalpy from a d(ρY_k) increment.
         *  dRhoYK[0..nTransported-1] are d(ρY_k)_code, rhoInc = d(ρ)_code.
         *  The dependent species (N2) contribution is absorbed automatically. */
        double mixtureFormationRhoEIncrement(double rhoInc, const double *dRhoYK, int nTransported) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;

        std::string mechanismFile_;
        std::string phaseName_;

        mutable std::vector<double> bufY_;  ///< Mass-fractions work buffer (per-instance).
        mutable std::vector<double> bufHf_; ///< Code-scaled formation enthalpies (per-instance, populated once).
    };

} // namespace DNDS::Euler::Chemistry
