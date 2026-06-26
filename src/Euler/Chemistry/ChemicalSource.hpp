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
#include <ostream>
#include <fmt/format.h>

namespace DNDS::Euler::Chemistry
{

    struct ConstSpeciesBufferView;

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
        operator ConstSpeciesBufferView() const;
    };

    /// Const version for input arrays.
    struct ConstSpeciesBufferView
    {
        const double *data = nullptr;
        int nSpecies = 0;

        double operator[](int i) const { return data[i]; }
    };

} // namespace Chemistry

template <>
struct fmt::formatter<DNDS::Euler::Chemistry::ConstSpeciesBufferView>
{
    char type = 'g';
    int precision = 6;
    std::string fmtSpec = "{:.6g}";

    auto parse(fmt::format_parse_context &ctx)
    {
        auto it = ctx.begin(), end = ctx.end();
        while (it != end && *it != '}')
        {
            switch (*it)
            {
            case 'e':
            case 'E':
            case 'f':
            case 'F':
            case 'g':
            case 'G':
                type = *it++;
                break;
            case '.':
            {
                it++;
                std::string v;
                while (it != end && *it >= '0' && *it <= '9')
                    v.push_back(*it++);
                if (!v.empty())
                    precision = std::stoi(v);
                break;
            }
            default:
                ++it;
            }
        }
        fmtSpec = fmt::format(FMT_STRING("{{:.{}{}}}"), precision, type);
        return it;
    }

    auto format(const DNDS::Euler::Chemistry::ConstSpeciesBufferView &Y,
                fmt::format_context &ctx) const
    {
        auto out = ctx.out();
        fmt::format_to(out, "[");
        for (int k = 0; k < Y.nSpecies; ++k)
        {
            if (k)
                fmt::format_to(out, ", ");
            fmt::format_to(out, fmtSpec, Y[k]);
        }
        return fmt::format_to(out, "]");
    }
};

namespace DNDS::Euler::Chemistry
{

    inline SpeciesBufferView::operator ConstSpeciesBufferView() const
    {
        return {data, nSpecies};
    }

    /**
     * @brief Dense Jacobian view: Ns rows x nCols cols, column-major.
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
        /**
         * @param mechanismFile  Path to Cantera YAML mechanism.
         * @param phaseName      Phase name (empty = default).
         * @param U0             Reference velocity scale [m/s]; used internally for
         *                       code-unit conversion in productionRatesAndJacobian
         *                       and mixture base internal energy.
         * @param rho0           Reference density scale [kg/m³]; used internally for
         *                       code-unit conversion in productionRatesAndJacobian.
         * @param TBase          Base/reference temperature [K] for per-species
         *                       internal-energy offsets. If <= 0, the minimum
         *                       per-species Cantera temperature is used.
         * @param transportModel Requested transport model. Only mixture-averaged
         *                       transport is currently implemented.
         */
        explicit ChemicalSource(const std::string &mechanismFile,
                                const std::string &phaseName,
                                double U0, double rho0,
                                double TBase = 0.0,
                                std::string transportModel = "MixtureAveraged");
        ~ChemicalSource();

        // Non-copyable, movable
        ChemicalSource(const ChemicalSource &) = delete;
        ChemicalSource &operator=(const ChemicalSource &) = delete;
        ChemicalSource(ChemicalSource &&) noexcept;
        ChemicalSource &operator=(ChemicalSource &&) noexcept;

        /// Deep-clone for thread-safety: creates independent Cantera Solution objects
        /// from the same mechanism file (no shared state between instances).
        std::unique_ptr<ChemicalSource> clone() const;

        const std::string &mechanismFile() const { return mechanismFile_; }
        const std::string &phaseName() const { return phaseName_; }

        int nSpecies() const;
        int nReactions() const;
        const std::vector<std::string> &speciesNames() const;
        const std::vector<double> &molecularWeights() const;
        const std::vector<double> &speciesGasConstants() const;

        /** Reference velocity scale [m/s] used for code-unit conversion. */
        double velScale() const;
        /** Reference density scale [kg/m³] used for code-unit conversion. */
        double rhoScale() const;
        /** Configured transport model name. Only mixture-averaged is implemented. */
        const std::string &transportModel() const;
        bool isMixtureAveragedTransport() const;

        // ---- Mixture thermodynamic properties (via Cantera EOS) ----

        double mixtureR(ConstSpeciesBufferView Y) const;
        double mixtureCp(double T, ConstSpeciesBufferView Y, double p = 101325) const;
        double mixtureCv(double T, ConstSpeciesBufferView Y, double p = 101325) const;
        double mixtureGamma(double T, ConstSpeciesBufferView Y, double p = 101325) const;
        double speedOfSound(double T, ConstSpeciesBufferView Y, double p = 101325) const;

        /** Specific internal energy [J/kg] at (T, p, Y). */
        double mixtureIntEnergy(double T, ConstSpeciesBufferView Y, double p = 101325) const;

        /** Specific enthalpy [J/kg] at (T, p, Y). */
        double mixtureEnthalpy(double T, ConstSpeciesBufferView Y, double p = 101325) const;

        /** Specific entropy [J/(kg*K)] at (T, p, Y). */
        double mixtureEntropy(double T, ConstSpeciesBufferView Y, double p = 101325) const;

        /** Lower valid thermodynamic temperature bound [K] reported by Cantera. */
        double minTemperature() const;

        /** Temperature [K] used for base internal-energy offsets. */
        double baseTemperature() const;

        /** Print detailed mechanism info (species, reactions, base temperature, etc.) to stream. */
        void printInfo(std::ostream &os) const;

        /**
         * Solve T from specific internal energy u [J/kg] and specific volume v [m³/kg].
         * Uses Cantera setState_UV (Newton). Optional T_guess [K] as warm-start.
         * rtol is forwarded to Cantera's internal UV solve.
         */
        double temperatureFromUV(double u, double v,
                                 ConstSpeciesBufferView Y,
                                 double T_guess = 0,
                                 double rtol = 1e-12) const;

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
         * T, p, Y are physical SI units.
         * rho, rhoE, rhoU, rhoV, rhoW are code-scaled (÷ρ0, ÷ρ0·U0², etc.).
         * Scale factors U0 and ρ0 are stored at construction and used internally
         * for the ideal-gas dT/dU and dP/dU chain rules.
         * jacFlags = bitmask of JacobianFlags.
         */
        void productionRatesAndJacobian(double T, double p, double rho,
                                        double rhoE, double rhoU, double rhoV, double rhoW,
                                        int iEnergy,
                                        ConstSpeciesBufferView Y,
                                        SpeciesBufferView omega,
                                        JacobianBufferView dOmegadU,
                                        int jacFlags = 0) const;

        /**
         * Advance a constant-volume ideal-gas reactor with an affine species RHS:
         * dY/dtau = chemistryScale * chemistryYdot(Y) - Y / linearTime + constantTerm.
         * Inputs and outputs are physical SI units. @p Y and @p constantTerm are full
         * nSpecies vectors. The reactor uses Cantera's ReactorNet/CVODE internally.
         */
        void advanceAffineConstVolume(double &T, double rho,
                                      SpeciesBufferView Y,
                                      double chemistryScale,
                                      double linearTime,
                                      ConstSpeciesBufferView constantTerm,
                                      double advanceTime,
                                      double rtol = 1e-10,
                                      double atol = 1e-18,
                                      int maxOrder = 1,
                                      int maxSteps = 10000000) const;

        /**
         * Advance a constant-volume ideal-gas reactor with chemistry only.
         * Inputs and outputs are physical SI units. @p Y is a full nSpecies vector.
         */
        void advanceConstVolume(double &T, double rho,
                                SpeciesBufferView Y,
                                double chemistryScale,
                                double advanceTime,
                                double rtol = 1e-10,
                                double atol = 1e-18,
                                int maxOrder = 0,
                                int maxSteps = 10000000) const;

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

        /** Per-species base internal energies [J/kg] at T_base (constant, pre-cached). */
        void speciesBaseInternalEnergies(SpeciesBufferView eBase) const;

        /** Specific base internal energy per mass Σ Y_k·e_base,k [J/kg] (physical units). */
        double mixtureBaseInternalEnergy(ConstSpeciesBufferView Y) const;

        /** Whether the Cantera thermo phase uses an ideal-gas EOS. */
        bool isIdealGas() const;

        // ---- Caller-owned buffer helpers ----------------------------------------

        /** Compute mass fractions from conservative species densities into caller-owned storage. */
        void massFractions(double rho, const double *rhoYK, int nTransported, SpeciesBufferView Y) const;

        /** Return read-only per-species base internal energies in code units (e_base,k/U0²). */
        ConstSpeciesBufferView mixtureBaseInternalRhoESpecies() const;

        /** Code-scaled volumetric base internal energy: rho · Σ Y_k · e_base,k / U0².
         *  Uses internal velScale() for code-unit conversion. */
        double mixtureBaseInternalRhoE(double rho, ConstSpeciesBufferView Y) const;

        /** Linearized increment of code-scaled base internal energy from a d(ρY_k) increment.
         *  dRhoYK[0..nTransported-1] are d(ρY_k)_code, rhoInc = d(ρ)_code.
         *  The dependent species (N2) contribution is absorbed automatically. */
        double mixtureBaseInternalRhoEIncrement(double rhoInc, const double *dRhoYK, int nTransported) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;

        std::string mechanismFile_;
        std::string phaseName_;
    };

} // namespace DNDS::Euler::Chemistry
