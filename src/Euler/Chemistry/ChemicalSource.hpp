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

        /**
         * Solve T from specific internal energy u [J/kg] and specific volume v [m³/kg].
         * Uses Cantera setState_UV (Newton). Optional T_guess [K] as warm-start.
         */
        double temperatureFromUV(double u, double v,
                                 ConstSpeciesBufferView Y,
                                 double T_guess = 0) const;

        // ---- Kinetics ----

        /** Net production rates ω_i [kmol/m³/s]. omega must have nSpecies elements. */
        void productionRates(double T, double p,
                             ConstSpeciesBufferView Y,
                             SpeciesBufferView omega) const;

        /**
         * Production rates AND Jacobian ∂ω/∂U.
         *   U = [ρ, ρu, ρv, ρw, ρE, ρY_0..ρY_{Ns-2}]
         * dOmegadU: Ns × nVars, column-major.
         */
        void productionRatesAndJacobian(double T, double p, double rho,
                                        ConstSpeciesBufferView Y,
                                        SpeciesBufferView omega,
                                        JacobianBufferView dOmegadU) const;

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

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace DNDS::Euler::Chemistry
