/** @file Gas.hpp
 *  @brief Ideal-gas Riemann solvers, flux functions, and thermodynamic utilities
 *         for the compressible Euler / Navier-Stokes equations.
 *
 *  Provides:
 *  - Right / left eigenvector matrices for the 1-D Euler system.
 *  - Conservative ↔ primitive variable conversions for an ideal gas.
 *  - Inviscid flux evaluation in Cartesian and face-normal directions (single
 *    and batched variants).
 *  - Roe, HLLC, and HLLEP approximate Riemann solvers with multiple entropy
 *    fix strategies selectable at compile time via the @c eigScheme template
 *    parameter, and at run time through RiemannSolverType dispatch.
 *  - Viscous (Navier-Stokes) flux with Sutherland viscosity, optional QCR
 *    correction, and adiabatic wall treatment.
 *
 *  All functions are templated on the spatial dimension @c dim (2 or 3) and
 *  accept Eigen expression types so that fixed-size and dynamic vectors /
 *  matrices are handled without copies.
 */
#pragma once
#include "DNDS/Defines.hpp"
#include "DNDS/IdealGasPhysics.hpp"

#include "DNDS/Serializer/JsonUtil.hpp"
#include "DNDS/Config/ConfigEnum.hpp"
#include <fmt/core.h>

namespace DNDS::Euler::Gas
{
    using tVec = Eigen::Vector3d;   ///< Convenience alias for 3-D real vector.
    using tVec2 = Eigen::Vector2d;  ///< Convenience alias for 2-D real vector.

    /// Maximum column count for batched (SIMD-style) Riemann solver evaluation.
    static const int MaxBatch = 16;

    // Shared constants for Riemann solver entropy fixes and low-dissipation schemes.
    // Delegated from DNDS::IdealGas.
    static constexpr real kScaleHartenYee = IdealGas::kScaleHartenYee; ///< Base Harten-Yee entropy-fix threshold (0.05).
    static constexpr real kScaleLD = IdealGas::kScaleLD;               ///< Low-dissipation speed-of-sound scaling (0.2).
    static constexpr real kScaleHFix = IdealGas::kScaleHFix;           ///< H-correction transverse-wave scaling (0.25).

    /**
     * @brief Selects the approximate Riemann solver and its entropy-fix variant.
     *
     * | Enumerator | Solver / fix scheme                                          |
     * |------------|--------------------------------------------------------------|
     * | Roe        | Standard Roe with Harten-Yee entropy fix (eigScheme 0).      |
     * | HLLC       | Harten-Lax-van Leer-Contact (known accuracy issues, see IV). |
     * | HLLEP      | HLL with Enhanced Pressure (type 0).                         |
     * | HLLEP_V1   | HLLEP variant 1 (type 1).                                   |
     * | Roe_M1     | Roe + cLLF entropy fix (eigScheme 1).                        |
     * | Roe_M2     | Roe + Lax-Friedrichs / vanilla Lax (eigScheme 2, early exit).|
     * | Roe_M3     | Roe + LD (low-dissipation) Roe fix (eigScheme 3).            |
     * | Roe_M4     | Roe + ID (intermediate-dissipation) Roe fix (eigScheme 4).   |
     * | Roe_M5     | Roe + LD cLLF fix (eigScheme 5).                             |
     * | Roe_M6     | Roe + H-correction only (eigScheme 6).                       |
     * | Roe_M7     | Roe + Harten-Yee fix only, no H-correction (eigScheme 7).    |
     * | Roe_M8     | Roe + H-correction + Harten-Yee fix (eigScheme 8).           |
     * | Roe_M9     | Reserved (eigScheme 9, currently asserts false).              |
     */
    enum RiemannSolverType
    {
        UnknownRS = 0,
        Roe = 1,
        HLLC = 2,
        HLLEP = 3,
        HLLEP_V1 = 21,
        Roe_M1 = 11,
        Roe_M2 = 12,
        Roe_M3 = 13,
        Roe_M4 = 14,
        Roe_M5 = 15,
        Roe_M6 = 16,
        Roe_M7 = 17,
        Roe_M8 = 18,
        Roe_M9 = 19,
    };

    DNDS_DEFINE_ENUM_JSON(
        RiemannSolverType,
        {
            {UnknownRS, "UnknownRS"},
            {Roe, "Roe"},
            {HLLC, "HLLC"},
            {HLLEP, "HLLEP"},
            {HLLEP_V1, "HLLEP_V1"},
            {Roe_M1, "Roe_M1"},
            {Roe_M2, "Roe_M2"},
            {Roe_M3, "Roe_M3"},
            {Roe_M4, "Roe_M4"},
            {Roe_M5, "Roe_M5"},
            {Roe_M6, "Roe_M6"},
            {Roe_M7, "Roe_M7"},
            {Roe_M8, "Roe_M8"},
            {Roe_M9, "Roe_M9"},
        })

    /**
     * @brief Fills the right eigenvector matrix for the 1-D Euler system in the
     *        x-direction.
     *
     * The matrix has (dim+2) columns, one per characteristic wave:
     *   - column 0:            left-running acoustic wave  (λ = u − a)
     *   - column 1:            entropy wave                (λ = u)
     *   - columns 2..dim:      shear waves                 (λ = u)
     *   - column dim+1:        right-running acoustic wave (λ = u + a)
     *
     * @warning @p ReV must be pre-allocated to size (dim+2)×(dim+2).
     *
     * @tparam dim   Spatial dimension (2 or 3).
     * @tparam TVec  Velocity vector type (Eigen expression, length @c dim).
     * @tparam TeV   Eigenvector matrix type (Eigen expression, (dim+2)×(dim+2)).
     * @param velo  Roe-averaged (or cell-centred) velocity vector.
     * @param Vsqr  Squared velocity magnitude (= velo.squaredNorm()).
     * @param H     Total specific enthalpy.
     * @param a     Speed of sound.
     * @param[out] ReV  Right eigenvector matrix, overwritten on output.
     */
    template <int dim = 3, class TVec, class TeV>
    inline void EulerGasRightEigenVector(const TVec &velo, real Vsqr, real H, real a, TeV &ReV)
    {
        ReV.setZero();
        ReV(0, {0, 1, dim + 1}).setConstant(1);
        ReV(Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>), Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>))
            .diagonal()
            .setConstant(1);

        ReV(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>), {0, 1, dim + 1}).colwise() = velo;
        ReV(1, 0) -= a;
        ReV(1, dim + 1) += a;

        // Last Row
        ReV(dim + 1, 0) = H - velo(0) * a;
        ReV(dim + 1, dim + 1) = H + velo(0) * a;
        ReV(dim + 1, 1) = 0.5 * Vsqr;

        ReV(dim + 1, Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>)) =
            velo(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim - 1>));
    }

    /**
     * @brief Fills the left eigenvector matrix (inverse of the right eigenvector
     *        matrix) for the 1-D Euler system in the x-direction.
     *
     * Satisfies LeV * ReV = I for the (dim+2)×(dim+2) system.
     *
     * @warning @p LeV must be pre-allocated to size (dim+2)×(dim+2).
     *
     * @tparam dim   Spatial dimension (2 or 3).
     * @tparam TVec  Velocity vector type.
     * @tparam TeV   Eigenvector matrix type.
     * @param velo   Roe-averaged (or cell-centred) velocity vector.
     * @param Vsqr   Squared velocity magnitude.
     * @param H      Total specific enthalpy.
     * @param a      Speed of sound.
     * @param gamma  Ratio of specific heats (γ).
     * @param[out] LeV  Left eigenvector matrix, overwritten on output.
     */
    template <int dim = 3, class TVec, class TeV>
    inline void EulerGasLeftEigenVector(const TVec &velo, real Vsqr, real H, real a, real gamma, TeV &LeV)
    {
        LeV.setZero();
        real gammaBar = gamma - 1;
        LeV(0, 0) = H + a / gammaBar * (velo(0) - a);
        LeV(0, 1) = -velo(0) - a / gammaBar;
        LeV(0, Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>)) =
            -velo(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim - 1>));
        LeV(0, dim + 1) = 1;

        LeV(1, 0) = -2 * H + 4 / gammaBar * (a * a);
        LeV(1, Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) = velo.transpose() * 2;
        LeV(1, dim + 1) = -2;

        LeV(Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>), 0) =
            velo(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim - 1>)) * (-2 * (a * a) / gammaBar);
        LeV(Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>), Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>))
            .diagonal()
            .setConstant(2 * (a * a) / gammaBar);

        LeV(dim + 1, 0) = H - a / gammaBar * (velo(0) + a);
        LeV(dim + 1, 1) = -velo(0) + a / gammaBar;
        LeV(dim + 1, Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>)) =
            -velo(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim - 1>));
        LeV(dim + 1, dim + 1) = 1;

        LeV *= gammaBar / (2 * a * a);
    }

    /// @brief Thin wrapper delegating to IdealGas::IdealGasThermal.
    inline void IdealGasThermal(
        real E, real rho, real vSqr, real gamma, real &p, real &asqr, real &H)
    {
        IdealGas::IdealGasThermal(E, rho, vSqr, gamma, p, asqr, H);
    }

    /**
     * @brief Pre-computed Roe-averaged quantities shared by all Riemann solvers.
     *
     * Holds the Lm/Rm thermodynamic state and the Roe averages derived from them.
     * Computed once by ComputeRoePreamble(), consumed by HLLEPFlux, HLLCFlux,
     * and RoeFlux.
     */
    template <int dim>
    struct RoePreamble
    {
        using TVec = Eigen::Vector<real, dim>;
        // L/R mean-state thermodynamics
        TVec veloLm, veloRm;                   ///< Left/right mean-state velocity vectors.
        real asqrLm, asqrRm, pLm, pRm, HLm, HRm; ///< Speed-of-sound², pressure, and enthalpy for L/R mean states.
        real vsqrLm, vsqrRm;                   ///< Squared velocity magnitudes for L/R mean states.
        // Roe averages
        TVec veloRoe;                           ///< Roe-averaged velocity vector.
        real sqrtRhoLm, sqrtRhoRm;              ///< √ρ for L/R states (used in Roe weighting).
        real vsqrRoe, HRoe, asqrRoe, rhoRoe, aRoe; ///< Roe-averaged |v|², H, a², ρ, and speed of sound.
    };

    /**
     * @brief Compute Roe-averaged quantities from mean-state L/R vectors.
     *
     * Extracts velocities, calls IdealGasThermal for both sides, then computes
     * the Roe-averaged velocity, enthalpy, and speed of sound.
     *
     * @param ULm  Left mean-state conservative vector
     * @param URm  Right mean-state conservative vector
     * @param gamma  Ratio of specific heats
     * @param dumpInfo  Callable invoked before assertion on invalid asqrRoe
     * @return RoePreamble<dim> with all fields populated
     */
    template <int dim = 3, typename TULm, typename TURm, typename TFdumpInfo>
    RoePreamble<dim> ComputeRoePreamble(const TULm &ULm, const TURm &URm,
                                        real gamma, const TFdumpInfo &dumpInfo)
    {
        RoePreamble<dim> rp;

        rp.veloLm = (ULm(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / ULm(0)).matrix();
        rp.veloRm = (URm(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / URm(0)).matrix();
        rp.vsqrLm = rp.veloLm.squaredNorm();
        rp.vsqrRm = rp.veloRm.squaredNorm();
        IdealGasThermal(ULm(dim + 1), ULm(0), rp.vsqrLm, gamma, rp.pLm, rp.asqrLm, rp.HLm);
        IdealGasThermal(URm(dim + 1), URm(0), rp.vsqrRm, gamma, rp.pRm, rp.asqrRm, rp.HRm);

        rp.sqrtRhoLm = std::sqrt(ULm(0));
        rp.sqrtRhoRm = std::sqrt(URm(0));

        rp.veloRoe = (rp.sqrtRhoLm * rp.veloLm + rp.sqrtRhoRm * rp.veloRm) / (rp.sqrtRhoLm + rp.sqrtRhoRm);
        rp.vsqrRoe = rp.veloRoe.squaredNorm();
        rp.HRoe = (rp.sqrtRhoLm * rp.HLm + rp.sqrtRhoRm * rp.HRm) / (rp.sqrtRhoLm + rp.sqrtRhoRm);
        rp.asqrRoe = (gamma - 1) * (rp.HRoe - 0.5 * rp.vsqrRoe);
        rp.rhoRoe = rp.sqrtRhoLm * rp.sqrtRhoRm;

        if (!(rp.asqrRoe > 0))
        {
            dumpInfo();
        }
        DNDS_assert(rp.asqrRoe > 0);
        rp.aRoe = std::sqrt(rp.asqrRoe);

        return rp;
    }

    /**
     * @brief Converts conservative variables to primitive variables for an
     *        ideal gas.
     *
     * Conservative layout:  U = [ρ, ρu, ρv, (ρw,) E, ...]
     * Primitive layout: prim = [ρ, u, v, (w,) p, ...]
     *
     * The primitive variable at index I4 = dim+1 stores **pressure** (Euler
     * module convention), not temperature.
     *
     * @tparam dim    Spatial dimension (2 or 3).
     * @tparam TCons  Conservative vector type (Eigen expression, length ≥ dim+2).
     * @tparam TPrim  Primitive vector type (Eigen expression, same size as TCons).
     * @param  U      Conservative state vector (input).
     * @param[out] prim  Primitive state vector (output).
     * @param  gamma  Ratio of specific heats (γ).
     */
    template <int dim = 3, class TCons, class TPrim>
    inline void IdealGasThermalConservative2Primitive(
        const TCons &U, TPrim &prim, real gamma)
    {
        prim = U / U(0);
        real vSqr = (U(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) / U(0)).squaredNorm();
        real rho = U(0);
        real E = U(1 + dim);
        real p = (gamma - 1) * (E - rho * 0.5 * vSqr);
        prim(0) = rho;
        prim(1 + dim) = p;
        DNDS_assert(rho > 0);
    }

    /**
     * @brief Converts primitive variables to conservative variables for an
     *        ideal gas.
     *
     * Inverse of IdealGasThermalConservative2Primitive().
     *
     * @tparam dim    Spatial dimension (2 or 3).
     * @tparam TCons  Conservative vector type.
     * @tparam TPrim  Primitive vector type.
     * @param  prim   Primitive state vector (input).
     * @param[out] U  Conservative state vector (output).
     * @param  gamma  Ratio of specific heats (γ).
     */
    template <int dim = 3, class TCons, class TPrim>
    inline void IdealGasThermalPrimitive2Conservative(
        const TPrim &prim, TCons &U, real gamma)
    {
        U = prim * prim(0);
        real vSqr = prim(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).squaredNorm();
        real rho = prim(0);
        real p = prim(dim + 1);
        real E = p / (gamma - 1) + (rho * 0.5 * vSqr);
        U(0) = rho;
        U(dim + 1) = E;
        DNDS_assert(rho > 0);
    }

    /**
     * @brief Computes total (stagnation) pressure p0 and temperature T0 from
     *        a primitive state using isentropic relations.
     *
     * Uses p0 = p * (1 + (γ−1)/2 · M²)^(γ/(γ−1)) and T0 = T * (1 + (γ−1)/2 · M²).
     *
     * @tparam dim    Spatial dimension (2 or 3).
     * @tparam TPrim  Primitive vector type.
     * @param  prim   Primitive state [ρ, u, v, (w,) p, ...].
     * @param  gamma  Ratio of specific heats (γ).
     * @param  rg     Specific gas constant R_gas = Cp − Cv.
     * @return std::tuple<real,real>  (p0, T0).
     */
    template <int dim = 3, class TPrim>
    std::tuple<real, real> IdealGasThermalPrimitiveGetP0T0(
        const TPrim &prim, real gamma, real rg)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto I4 = dim + 1;
        real T = prim(I4) / (prim(0) * rg + verySmallReal);
        real vsqr = prim(Seq123).squaredNorm();
        real asqr = gamma * prim(I4) / prim(0);
        real Msqr = vsqr / (asqr + verySmallReal);
        real p0 = std::pow(1 + (gamma - 1) * 0.5 * Msqr, gamma / (gamma - 1)) * prim(I4);
        real T0 = (1 + (gamma - 1) * 0.5 * Msqr) * T;
        return std::make_tuple(p0, T0);
    }

    /**
     * @brief Computes the inviscid (Euler) flux in the x-direction for a
     *        moving grid.
     *
     * The flux vector is written to the first (dim+2) entries of @p F:
     *   F = [ρ(u−ug), ρu(u−ug)+p, ρv(u−ug), (ρw(u−ug),) (E+p)u − E·ug]
     * where ug = vg(0) is the grid velocity in the x-direction.
     *
     * @note Passive-scalar (RANS) entries beyond index dim+1 are **not** filled.
     *
     * @tparam dim     Spatial dimension (2 or 3).
     * @param  U       Conservative state vector.
     * @param  velo    Fluid velocity vector.
     * @param  vg      Grid velocity vector (for ALE / moving-mesh formulations).
     * @param  p       Pressure.
     * @param[out] F   Flux vector (first dim+2 entries overwritten).
     */
    template <int dim = 3, typename TU, typename TF, class TVec, class TVecVG>
    inline void GasInviscidFlux(const TU &U, const TVec &velo, const TVecVG &vg, real p, TF &F)
    {
        F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = U(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) * (velo(0) - vg(0)); // note that additional flux are unattended!
        F(1) += p;
        F(dim + 1) += velo(0) * p;
        // original form: F(dim + 1) += (velo(0) - vg(0)) * p + vg(0) * p;
    }

    /**
     * @brief Computes the inviscid flux projected onto an arbitrary face normal
     *        @p n, accounting for grid motion @p vg.
     *
     * This is the main face-flux function used during spatial discretisation.
     * The normal velocity is V_n = (velo − vg) · n.
     *
     * @note Passive-scalar entries beyond index dim+1 are **not** filled.
     *
     * @tparam dim  Spatial dimension.
     * @param  U    Conservative state vector.
     * @param  velo Fluid velocity vector.
     * @param  vg   Grid velocity vector.
     * @param  n    Outward face-normal vector (unit or area-weighted).
     * @param  p    Pressure.
     * @param[out] F  Flux vector (first dim+2 entries overwritten).
     */
    template <int dim = 3, typename TU, typename TF, class TVec, class TVecN, class TVecVG>
    inline void GasInviscidFlux_XY(const TU &U, const TVec &velo, const TVecVG &vg, const TVecN &n, real p, TF &F)
    {
        F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = U(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) * (velo - vg).dot(n); // note that additional flux are unattended!
        F(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) += p * n;
        F(dim + 1) += velo.dot(n) * p;
        // original form: F(dim + 1) += (velo(0) - vg(0)) * p + vg(0) * p;
    }

    /**
     * @brief Batched x-direction inviscid flux for column-major state matrices.
     *
     * Each column of @p U, @p velo, @p vg represents one face in the batch;
     * @p p is a row-vector of pressures. Output columns of @p F are filled
     * independently.
     *
     * @tparam dim  Spatial dimension.
     * @param  U    Conservative states   [(dim+2) × nBatch].
     * @param  velo Velocity vectors      [dim × nBatch].
     * @param  vg   Grid velocities       [dim × nBatch].
     * @param  p    Pressures             [1 × nBatch].
     * @param[out] F  Flux matrix         [(dim+2) × nBatch].
     */
    template <int dim = 3, typename TU, typename TF, class TVec, class TVecVG, class TP>
    inline void GasInviscidFlux_Batch(const TU &U, const TVec &velo, const TVecVG &vg, TP &&p, TF &F)
    {
        F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>), EigenAll) = U(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>), EigenAll).array().rowwise() * (velo(0, EigenAll) - vg(0, EigenAll)).array(); // note that additional flux are unattended!
        F(1, EigenAll).array() += p.array();
        F(dim + 1, EigenAll).array() += velo(0, EigenAll).array() * p.array();
        // original form: F(dim + 1) += (velo(0) - vg(0)) * p + vg(0) * p;
    }

    /**
     * @brief Batched face-normal inviscid flux for column-major state matrices.
     *
     * Each column represents one face in the batch. The flux at each column is
     * projected onto the corresponding column of the normal matrix @p n.
     *
     * @tparam dim  Spatial dimension.
     * @param  U    Conservative states   [(dim+2) × nBatch].
     * @param  velo Velocity vectors      [dim × nBatch].
     * @param  vg   Grid velocities       [dim × nBatch].
     * @param  n    Face normals           [dim × nBatch].
     * @param  p    Pressures             [1 × nBatch].
     * @param[out] F  Flux matrix         [(dim+2) × nBatch].
     */
    template <int dim = 3, typename TU, typename TF, class TVec, class TVecVG, class TVecN, class TP>
    inline void GasInviscidFlux_XY_Batch(const TU &U, const TVec &velo, const TVecVG &vg, const TVecN &n, TP &&p, TF &F)
    {
        auto vn = (velo.array() * n.array()).colwise().sum();
        auto vgn = (vg.array() * n.array()).colwise().sum();
        F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>), EigenAll) = U(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>), EigenAll).array().rowwise() * (vn - vgn).array(); // note that additional flux are unattended!
        F(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>), EigenAll).array() += n.array().rowwise() * p.array();
        F(dim + 1, EigenAll).array() += vn * p.array();
        // original form: F(dim + 1) += (velo(0) - vg(0)) * p + vg(0) * p;
    }

    /**
     * @brief Computes velocity and pressure increments from a conservative-state
     *        increment, used for the Lax-flux Jacobian computation.
     *
     * Given a state U and its increment dU, computes
     *   dVelo = d(momentum/ρ)  and  dp = (γ−1)(dE − ½(dMomentum·v + momentum·dv)).
     *
     * @tparam dim  Spatial dimension.
     * @param  U     Conservative state vector.
     * @param  dU    Conservative state increment.
     * @param  velo  Velocity vector (= U[1..dim] / U[0]).
     * @param  gamma Ratio of specific heats (γ).
     * @param[out] dVelo  Velocity increment (output).
     * @param[out] dp     Pressure increment (output).
     */
    template <int dim = 3, typename TU, class TVec>
    inline void IdealGasUIncrement(const TU &U, const TU &dU, const TVec &velo, real gamma, TVec &dVelo, real &dp)
    {
        dVelo = (dU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) -
                 U(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) / U(0) * dU(0)) /
                U(0);
        dp = (gamma - 1) * (dU(dim + 1) -
                            0.5 * (dU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).dot(velo) +
                                   U(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).dot(dVelo)));
    } // For Lax-Flux jacobian

    /**
     * @brief Computes the increment of the facial inviscid flux from state,
     *        velocity, and pressure increments.
     *
     * Used in the implicit time-stepping Jacobian (e.g. LU-SGS).
     * Writes the full flux vector @p F including passive-scalar entries
     * beyond index dim+1 (valid for RANS and extra-equation models).
     *
     * \note now this function writes to the whole F, assuming SeqI52Last part is passive scalar
     * this is valid for RANS and EX
     *
     * @tparam dim  Spatial dimension.
     * @param  U        Conservative state vector.
     * @param  dU       Conservative state increment.
     * @param  unitNorm Face unit-normal vector.
     * @param  velo     Velocity vector.
     * @param  dVelo    Velocity increment (from IdealGasUIncrement()).
     * @param  vg       Grid velocity vector.
     * @param  dp       Pressure increment (from IdealGasUIncrement()).
     * @param  p        Pressure.
     * @param[out] F    Incremental facial flux (all entries written).
     */
    template <int dim = 3, typename TU, typename TF, class TVec, class TVecVG>
    inline void GasInviscidFluxFacialIncrement(const TU &U, const TU &dU,
                                               const TVec &unitNorm,
                                               const TVecVG &velo, const TVec &dVelo, const TVec &vg,
                                               real dp, real p,
                                               TF &F)
    {
        real vn = velo.dot(unitNorm);
        real dvn = dVelo.dot(unitNorm);
        F(0) = dU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).dot(unitNorm);
        F(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) =
            dU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) * vn +
            U(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) * dvn + unitNorm * dp;
        F(dim + 1) = (dU(dim + 1) + dp) * vn + (U(dim + 1) + p) * dvn;
        static const auto SeqI52Last = Eigen::seq(Eigen::fix<dim + 2>, EigenLast);
        if constexpr (TU::RowsAtCompileTime == Eigen::Dynamic || TU::RowsAtCompileTime > dim + 2)
            F(SeqI52Last) = dU(SeqI52Last) * vn + U(SeqI52Last) * dvn;
        F -= dU * vg.dot(unitNorm);
    }

    /**
     * @brief Convenience wrapper that computes the right eigenvector matrix
     *        directly from a conservative state vector U.
     *
     * Extracts velocity and thermodynamic quantities from U, then delegates to
     * EulerGasRightEigenVector().
     *
     * @tparam dim  Spatial dimension.
     * @param  U      Conservative state vector.
     * @param  gamma  Ratio of specific heats (γ).
     * @return Eigen::Matrix<real, dim+2, dim+2>  Right eigenvector matrix.
     */
    template <int dim = 3, typename TU>
    inline auto IdealGas_EulerGasRightEigenVector(const TU &U, real gamma)
    {
        DNDS_assert(U(0) > 0);
        Eigen::Matrix<real, dim + 2, dim + 2> ReV;
        Eigen::Vector<real, dim> velo =
            (U(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / U(0)).matrix();
        real vsqr = velo.squaredNorm();
        real asqr, p, H;
        IdealGasThermal(U(dim + 1), U(0), vsqr, gamma, p, asqr, H);
        DNDS_assert(asqr >= 0);
        EulerGasRightEigenVector<dim>(velo, vsqr, H, std::sqrt(asqr), ReV);
        return ReV;
    }

    /**
     * @brief Convenience wrapper that computes the left eigenvector matrix
     *        directly from a conservative state vector U.
     *
     * Extracts velocity and thermodynamic quantities from U, then delegates to
     * EulerGasLeftEigenVector().
     *
     * @tparam dim  Spatial dimension.
     * @param  U      Conservative state vector.
     * @param  gamma  Ratio of specific heats (γ).
     * @return Eigen::Matrix<real, dim+2, dim+2>  Left eigenvector matrix.
     */
    template <int dim = 3, typename TU>
    inline auto IdealGas_EulerGasLeftEigenVector(const TU &U, real gamma)
    {
        DNDS_assert(U(0) > 0);
        Eigen::Matrix<real, dim + 2, dim + 2> LeV;
        Eigen::Vector<real, dim> velo =
            (U(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / U(0)).matrix();
        real vsqr = velo.squaredNorm();
        real asqr, p, H;
        IdealGasThermal(U(dim + 1), U(0), vsqr, gamma, p, asqr, H);
        DNDS_assert(asqr >= 0);
        EulerGasLeftEigenVector<dim>(velo, vsqr, H, std::sqrt(asqr), gamma, LeV);
        return LeV;
    }

    /**
     * @brief HLLEP (HLL with Enhanced Pressure) approximate Riemann solver for
     *        an ideal gas on a moving grid.
     *
     * Uses Roe-decomposed wave strengths combined with HLL wave-speed estimates.
     * When @p type = 0, computes the standard HLLEP flux; when @p type = 1,
     * computes the HLLEP_V1 variant which uses modified eigenvalue scaling
     * factors δ1, δ2, δ3 instead of the HLL blending formula.
     *
     * Left/right face states (@p UL, @p UR) are used for the actual flux and
     * jump, while the mean states (@p ULm, @p URm) are used for Roe averaging
     * (these may differ under MUSCL or WENO reconstruction).
     *
     * @tparam dim   Spatial dimension (2 or 3).
     * @tparam type  0 = HLLEP, 1 = HLLEP_V1.
     * @param  UL, UR      Left/right conservative face states.
     * @param  ULm, URm    Left/right mean-state conservative vectors for Roe averaging.
     * @param  vg           Grid velocity vector.
     * @param  n            Face-normal vector (unit or area-weighted).
     * @param  gamma        Ratio of specific heats (γ).
     * @param[out] F        Numerical flux (first dim+2 entries written).
     * @param  dLambda      H-correction transverse eigenvalue estimate from
     *                      neighbouring faces.
     * @param  fixScale     User-configurable scaling factor for entropy fix
     *                      (from settings.rsFixScale).
     * @param  dumpInfo     Callable invoked before assertion on invalid state
     *                      (negative density, NaN, etc.).
     * @param[out] lam0     Absolute eigenvalue |V_n − a| after entropy fixing.
     * @param[out] lam123   Absolute eigenvalue |V_n| after entropy fixing.
     * @param[out] lam4     Absolute eigenvalue |V_n + a| after entropy fixing.
     */
    // #define DNDS_GAS_HLLEP_USE_V1
    template <int dim = 3, int type = 0,
              typename TUL, typename TUR,
              typename TULm, typename TURm, typename TVecVG, typename TVecN,
              typename TF, typename TFdumpInfo>
    void HLLEPFlux_IdealGas(const TUL &UL, const TUR &UR, const TULm &ULm, const TURm &URm,
                            const TVecVG &vg, const TVecN &n,
                            real gamma, TF &F, real dLambda, real fixScale,
                            const TFdumpInfo &dumpInfo, real &lam0, real &lam123, real &lam4)
    {
        using TVec = Eigen::Vector<real, dim>;

        if (!(UL(0) > 0 && UR(0) > 0))
        {
            dumpInfo();
        }
        DNDS_assert(UL(0) > 0 && UR(0) > 0);
        TVec veloL = (UL(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / UL(0)).matrix();
        TVec veloR = (UR(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / UR(0)).matrix();

        real asqrL, asqrR, pL, pR, HL, HR;
        real vsqrL = veloL.squaredNorm();
        real vsqrR = veloR.squaredNorm();
        IdealGasThermal(UL(dim + 1), UL(0), vsqrL, gamma, pL, asqrL, HL);
        IdealGasThermal(UR(dim + 1), UR(0), vsqrR, gamma, pR, asqrR, HR);

        auto rp = ComputeRoePreamble<dim>(ULm, URm, gamma, dumpInfo);

        Eigen::Vector<real, dim + 2> FL, FR;
        GasInviscidFlux_XY<dim>(UL, veloL, vg, n, pL, FL);
        GasInviscidFlux_XY<dim>(UR, veloR, vg, n, pR, FR);

        real veloRoeN = rp.veloRoe.dot(n);
        real vgN = vg.dot(n);
        real veloRoe0 = veloRoeN - vgN;
        lam0 = std::abs(veloRoe0 - rp.aRoe);
        lam123 = std::abs(veloRoe0);
        lam4 = std::abs(veloRoe0 + rp.aRoe);
        real veloLm0 = (rp.veloLm - vg).dot(n);
        real veloRm0 = (rp.veloRm - vg).dot(n);

        Eigen::Vector<real, dim + 2> incU =
            UR(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) -
            UL(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)); //! not using m, for this is accuracy-limited!
        real incU123N = incU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).dot(n);

        TVec alpha23V = incU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) - incU(0) * rp.veloRoe;
        TVec alpha23VT = alpha23V - n * alpha23V.dot(n);
        real incU4b = incU(dim + 1) - alpha23VT.dot(rp.veloRoe);
        real alpha1 = (gamma - 1) / rp.asqrRoe *
                      (incU(0) * (rp.HRoe - veloRoeN * veloRoeN) +
                       veloRoeN * incU123N - incU4b);
        real alpha0 = (incU(0) * (veloRoeN + rp.aRoe) - incU123N - rp.aRoe * alpha1) / (2 * rp.aRoe);
        real alpha4 = incU(0) - (alpha0 + alpha1);

        // std::cout << alpha.transpose() << std::endl;
        // std::cout << std::endl;

        real SL = std::min(veloRoe0 - rp.aRoe, veloLm0 - std::sqrt(asqrL));
        real SR = std::max(veloRoe0 + rp.aRoe, veloRm0 + std::sqrt(asqrR));
        real UU = std::abs(veloRoe0);
        real dfix = rp.aRoe / (rp.aRoe + std::max(UU, dLambda * fixScale));
        real SP = std::max(SR, 0.0);
        real SM = std::min(SL, 0.0);

        if constexpr (type != 1) // ? HLLEP
        {
            Eigen::Vector<real, dim + 2> ReV1;
            {
                ReV1(0) = 1;
                ReV1(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) = rp.veloRoe;
                ReV1(dim + 1) = rp.vsqrRoe * 0.5;
            }
            real div = SP - SM;
            div += signP(div) * verySmallReal;
            // F = (SP * FL - SM * FR) / div + (SP * SM / div) * (UR - UL - dfix * ReVRoe(EigenAll, {1, 2, 3}) * alpha({1, 2, 3}));
            F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) =
                (SP * FL - SM * FR) / div +
                (SP * SM / div) *
                    (UR(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) -
                     UL(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) -
                     dfix * ReV1 * alpha1);
        }
        else
        {
            // ? HLLEP_V1
            // real delta1 = aRoe / (UU + aRoe + verySmallReal);
            // real delta1 = 0.5;
            real delta1 = dfix;
            real delta2 = 0.0;
            real delta3 = 0.0;

            lam123 = ((SP + SM) * (veloRoe0)-2.0 * (1.0 - delta1) * (SP * SM)) / (SP - SM);
            lam4 = ((SP + SM) * (veloRoe0 + rp.aRoe) - 2.0 * (1.0 - delta2) * (SP * SM)) / (SP - SM);
            lam0 = ((SP + SM) * (veloRoe0 - rp.aRoe) - 2.0 * (1.0 - delta3) * (SP * SM)) / (SP - SM);
            lam123 = std::abs(lam123);
            lam0 = std::abs(lam0);
            lam4 = std::abs(lam4);

            alpha0 *= lam0;
            alpha1 *= lam123;
            alpha23VT *= lam123;
            alpha4 *= lam4; // here becomes alpha_i * lam_i

            Eigen::Vector<real, dim + 2> incF;
            incF(0) = alpha0 + alpha1 + alpha4;
            incF(dim + 1) = (rp.HRoe - veloRoeN * rp.aRoe) * alpha0 + 0.5 * rp.vsqrRoe * alpha1 +
                            (rp.HRoe + veloRoeN * rp.aRoe) * alpha4 + alpha23VT.dot(rp.veloRoe);
            incF(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) =
                (rp.veloRoe - rp.aRoe * n) * alpha0 + (rp.veloRoe + rp.aRoe * n) * alpha4 +
                rp.veloRoe * alpha1 + alpha23VT;

            F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = (FL + FR) * 0.5 - 0.5 * incF;
        }
    }

    /**
     * @brief HLLC (Harten-Lax-van Leer-Contact) approximate Riemann solver for
     *        an ideal gas on a moving grid.
     *
     * @warning This solver has known accuracy issues (observed in the IV test
     *          problem). Prefer Roe or HLLEP for production runs.
     *
     * @tparam dim  Spatial dimension (2 or 3).
     * @param  UL, UR      Left/right conservative face states.
     * @param  ULm, URm    Left/right mean states for Roe averaging.
     * @param  vg           Grid velocity vector.
     * @param  n            Face-normal vector.
     * @param  gamma        Ratio of specific heats (γ).
     * @param[out] F        Numerical flux (first dim+2 entries written).
     * @param  dLambda      H-correction transverse eigenvalue estimate.
     * @param  fixScale     Entropy-fix scaling factor.
     * @param  dumpInfo     Debug dump callable for assertion failures.
     * @param[out] lam0     |V_n − a| (Roe-averaged, before HLLC wave selection).
     * @param[out] lam123   |V_n| (Roe-averaged).
     * @param[out] lam4     |V_n + a| (Roe-averaged).
     */
    template <int dim = 3, typename TUL, typename TUR, typename TULm, typename TURm,
              typename TVecVG, typename TVecN,
              typename TF, typename TFdumpInfo>
    void HLLCFlux_IdealGas_HartenYee(const TUL &UL, const TUR &UR, const TULm &ULm, const TURm &URm,
                                     const TVecVG &vg, const TVecN &n,
                                     real gamma, TF &F, real dLambda, real fixScale,
                                     const TFdumpInfo &dumpInfo, real &lam0, real &lam123, real &lam4)
    {
        //! warning: has accuracy issue (see IV test)
        using TVec = Eigen::Vector<real, dim>;

        if (!(UL(0) > 0 && UR(0) > 0))
        {
            dumpInfo();
        }
        DNDS_assert(UL(0) > 0 && UR(0) > 0);
        TVec veloL = (UL(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / UL(0)).matrix();
        TVec veloR = (UR(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / UR(0)).matrix();

        real asqrL, asqrR, pL, pR, HL, HR;
        real vsqrL = veloL.squaredNorm();
        real vsqrR = veloR.squaredNorm();
        IdealGasThermal(UL(dim + 1), UL(0), vsqrL, gamma, pL, asqrL, HL);
        IdealGasThermal(UR(dim + 1), UR(0), vsqrR, gamma, pR, asqrR, HR);

        auto rp = ComputeRoePreamble<dim>(ULm, URm, gamma, dumpInfo);

        Eigen::Vector<real, dim + 2> FL, FR;
        GasInviscidFlux_XY<dim>(UL, veloL, vg, n, pL, FL);
        GasInviscidFlux_XY<dim>(UR, veloR, vg, n, pR, FR);

        real veloRoeN = rp.veloRoe.dot(n);
        real vgN = vg.dot(n);
        real veloRoe0 = veloRoeN - vgN;
        lam0 = std::abs(veloRoe0 - rp.aRoe);
        lam123 = std::abs(veloRoe0);
        lam4 = std::abs(veloRoe0 + rp.aRoe);
        real veloLm0 = (rp.veloLm - vg).dot(n);
        real veloRm0 = (rp.veloRm - vg).dot(n);

        auto HLLCq = [&](real p, real pS)
        {
            real q = std::sqrt(1 + (gamma + 1) / 2 / gamma * (pS / p - 1));
            if (pS <= p)
                q = 1;
            return q;
        };
        real pS = 0.5 * (rp.pLm + rp.pRm) - 0.5 * (veloLm0 - veloRm0) * rp.rhoRoe * rp.aRoe;
        pS = std::max(0.0, pS);
        real SL = veloLm0 - std::sqrt(rp.asqrLm) * HLLCq(rp.pLm, pS);
        real SR = veloRm0 + std::sqrt(rp.asqrRm) * HLLCq(rp.pRm, pS);

        dLambda += verySmallReal;
        dLambda *= 2.0;

        // * E-Fix
        // SL += sign(SL) * std::exp(-std::abs(SL) / dLambda) * dLambda;
        // SR += sign(SR) * std::exp(-std::abs(SR) / dLambda) * dLambda;

        if (0 <= SL)
        {
            F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = FL;
            return;
        }
        if (SR <= 0)
        {
            F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = FR;
            return;
        }
        real SS = 0;
        real div = (UL(0) * (SL - veloLm0) - UR(0) * (SR - veloRm0));
        if (std::abs(div) > verySmallReal)
            SS = (pR - pL +
                  (UL(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).dot(n) - vgN * UL(0)) * (SL - veloLm0) -
                  (UR(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).dot(n) - vgN * UR(0)) * (SR - veloRm0)) /
                 div;
        //! is this right for moving mesh?
        Eigen::Vector<real, dim + 2> DS;
        DS.setZero();
        DS(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) = n;
        DS(dim + 1) = SS;
        // SS += sign(SS) * std::exp(-std::abs(SS) / dLambda) * dLambda;
        if (SS >= 0)
        {
            real div = SL - SS;
            if (std::abs(div) < verySmallReal)
                F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = FL;
            else
                F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) =
                    ((UL(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) * SL - FL) * SS +
                     DS * ((pL + UL(0) * (SL - veloLm0) * (SS - veloLm0)) * SL)) /
                    div;
        }
        else
        {
            real div = SR - SS;
            if (std::abs(div) < verySmallReal)
                F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = FR;
            else
                F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) =
                    ((UR(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) * SR - FR) * SS +
                     DS * ((pR + UR(0) * (SR - veloRm0) * (SS - veloRm0)) * SR)) /
                    div;
        }
    }

    /**
     * @brief Template-dispatched entropy-fix for Roe-type Riemann solvers.
     *
     * Modifies the absolute Roe eigenvalues (lam0, lam123, lam4) in-place to
     * prevent the sonic glitch and control numerical dissipation. The fix
     * strategy is selected at compile time via @p eigScheme:
     *
     * | eigScheme | Strategy                                                    |
     * |-----------|-------------------------------------------------------------|
     * | 0         | Harten-Yee (H2) — threshold = max(0, λ−λ_L, λ_R−λ) · s.   |
     * | 1         | cLLF — componentwise local Lax-Friedrichs.                  |
     * | 3         | LD Roe — low-dissipation (Fleischmann et al. 2020).          |
     * | 4         | ID Roe — intermediate-dissipation (Fleischmann et al. 2020). |
     * | 5         | LD cLLF — low-dissipation componentwise LLF.                |
     * | 6         | H-correction only (floor by dLambda · scaleHFix).           |
     * | 7         | Harten-Yee only, no H-correction.                           |
     * | 8         | H-correction + Harten-Yee combined.                         |
     *
     * @tparam eigScheme  Compile-time entropy-fix scheme selector (0–8).
     * @param  aL, aR     Left/right mean-state speed of sound.
     * @param  aAve       Roe-averaged speed of sound.
     * @param  uL, uR     Left/right normal velocities relative to grid.
     * @param  uAve       Roe-averaged normal velocity relative to grid.
     * @param  VL, VR     Left/right total velocity magnitudes relative to grid.
     * @param  VAve       Roe-averaged total velocity magnitude relative to grid.
     * @param  dLambda    H-correction transverse eigenvalue estimate.
     * @param  fixScale   User scaling factor applied to the base entropy-fix
     *                    constants (kScaleHartenYee, kScaleLD, kScaleHFix).
     * @param  incFScale  Incremental flux dissipation scale (settings.rsIncFScale),
     *                    controls overall dissipation level in the contact/shear
     *                    wave.
     * @param[in,out] lam0    |V_n − a| — modified in place.
     * @param[in,out] lam123  |V_n|     — modified in place.
     * @param[in,out] lam4    |V_n + a| — modified in place.
     */
    template <int eigScheme>
    void Roe_EntropyFixer(const real aL, const real aR, const real aAve,
                          const real uL, const real uR, const real uAve,
                          const real VL, const real VR, const real VAve, // V is magnitude of velo
                          real dLambda, real fixScale, real incFScale,
                          real &lam0, real &lam123, real &lam4)
    {
        const real scaleHartenYee = kScaleHartenYee * fixScale;
        const real scaleLD = kScaleLD * fixScale;
        const real scaleHFix = kScaleHFix * fixScale;
        static const real incFScaleA = 1.0;
        if constexpr (eigScheme == 0)
        {
            //*H2
            auto Flim = [=](real v, real lam, real lamL, real lamR)
            {
                const real thresH = std::max<real>({0., lam - lamL, lamR - lam}) * scaleHartenYee;
                if (v < thresH)
                    return (sqr(v) / thresH + thresH) * 0.5;
                else
                    return v;
            };

            lam0 = Flim(lam0 * incFScaleA, uAve - aAve, uL - aL, uR - aR);
            lam123 = Flim(lam123 * incFScale, uAve, uL, uR);
            lam4 = Flim(lam4 * incFScaleA, uAve + aAve, uL + aL, uR + aR);
            //*H2

            // lam0 = std::max(lam0, dLambda * scaleHFix);
            // lam123 = std::max(lam123, dLambda * scaleHFix);
            // lam4 = std::max(lam4, dLambda * scaleHFix);
        }
        else if constexpr (eigScheme == 1)
        {
            //* cLLF
            const real aLm = aL * incFScaleA;
            const real aRm = aR * incFScaleA;
            const real veloLm0 = uL;
            const real veloRm0 = uR;
            const real uLm = signTol(veloLm0, aLm * smallReal) * std::max(std::abs(veloLm0) * incFScale, aLm * scaleLD);
            const real uRm = signTol(veloRm0, aRm * smallReal) * std::max(std::abs(veloRm0) * incFScale, aRm * scaleLD);
            lam0 = std::max(std::abs(uLm - aLm), std::abs(uRm - aRm));
            lam123 = std::max(std::abs(uLm), std::abs(uRm));
            lam4 = std::max(std::abs(uLm + aLm), std::abs(uRm + aRm));
        }
        else if constexpr (eigScheme == 3)
        {
            //*LD, Roe_M
            /**
             * Nico Fleischmann, Stefan Adami, Xiangyu Y. Hu, Nikolaus A. Adams, A low dissipation method to cure the grid-aligned shock instability, 2020
             */
            const real LDthreshold = std::abs(uAve) / scaleLD;
            const real aRoeStar = std::min(LDthreshold, aAve * incFScaleA);
            lam0 = std::abs(uAve * incFScale - aRoeStar);
            lam4 = std::abs(uAve * incFScale + aRoeStar);
        }
        else if constexpr (eigScheme == 4)
        {
            //*ID, Roe_M
            /**
             * Nico Fleischmann, Stefan Adami, Xiangyu Y. Hu, Nikolaus A. Adams, A low dissipation method to cure the grid-aligned shock instability, 2020
             */
#ifdef USE_SIGN_MINUS_AT_ROE_M4_FLUX
            const real uStar = signM(uAve) * std::max(aAve * scaleLD, std::abs(uAve));
#else
            const real uStar = signTol(uAve, aAve * smallReal) * std::max(aAve * scaleLD, std::abs(uAve) * incFScale); //! why signM here?
#endif
            lam0 = std::abs(uStar - aAve * incFScaleA);
            lam123 = std::abs(uStar);
            lam4 = std::abs(uStar + aAve * incFScaleA);
        }
        else if constexpr (eigScheme == 5)
        {
            //*LD, cLLF_M
            /**
             * Nico Fleischmann, Stefan Adami, Xiangyu Y. Hu, Nikolaus A. Adams, A low dissipation method to cure the grid-aligned shock instability, 2020
             */
            const real veloLm0 = uL * incFScale;
            const real veloRm0 = uR * incFScale;
            const real aLm = std::min(aL * incFScaleA, std::abs(veloLm0) / scaleLD);
            const real aRm = std::min(aR * incFScaleA, std::abs(veloRm0) / scaleLD);
            lam0 = std::max(std::abs(veloLm0 - aLm), std::abs(veloRm0 - aRm));
            lam123 = std::max(std::abs(veloLm0), std::abs(veloRm0));
            lam4 = std::max(std::abs(veloLm0 + aLm), std::abs(veloRm0 + aRm));
            //*LD, cLLF_M
        }
        else if constexpr (eigScheme == 6) // simply H-correction
        {
            lam0 = std::max(std::abs(uAve * incFScale - aAve * incFScaleA), dLambda * scaleHFix);
            lam4 = std::max(std::abs(uAve * incFScale + aAve * incFScaleA), dLambda * scaleHFix);
            lam123 = std::max(std::abs(uAve * incFScale), dLambda * scaleHFix);
        }
        else if constexpr (eigScheme == 7) // simply Harten-Yee-fix
        {
            lam0 *= std::abs(uAve * incFScale - aAve * incFScaleA);
            lam4 *= std::abs(uAve * incFScale + aAve * incFScaleA);
            lam123 *= std::abs(uAve * incFScale);
            //*HY
            const real thresholdHartenYee = scaleHartenYee * (VAve + aAve);
            const real thresholdHartenYeeS = sqr(thresholdHartenYee);
            if (lam0 < thresholdHartenYee)
                lam0 = (sqr(lam0) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
            if (lam4 < thresholdHartenYee)
                lam4 = (sqr(lam4) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
            if (lam123 < thresholdHartenYee)
                lam123 = (sqr(lam123) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
            //*HY
        }
        else if constexpr (eigScheme == 8) // H-cor + Harten-Yee-fix
        {
            lam0 = std::max(std::abs(uAve * incFScale - aAve * incFScaleA), dLambda * scaleHFix);
            lam4 = std::max(std::abs(uAve * incFScale + aAve * incFScaleA), dLambda * scaleHFix);
            lam123 = std::max(std::abs(uAve * incFScale), dLambda * scaleHFix);

            //*HY
            const real thresholdHartenYee = scaleHartenYee * (VAve + aAve);
            const real thresholdHartenYeeS = sqr(thresholdHartenYee);
            if (lam0 < thresholdHartenYee)
                lam0 = (sqr(lam0) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
            if (lam4 < thresholdHartenYee)
                lam4 = (sqr(lam4) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
            if (lam123 < thresholdHartenYee)
                lam123 = (sqr(lam123) + thresholdHartenYeeS) / (2 * thresholdHartenYee);
            //*HY
        }
        else
        {
            DNDS_assert(false);
        }
    }

    /**
     * @brief Core Roe approximate Riemann solver with a selectable entropy-fix
     *        scheme for an ideal gas on a moving grid.
     *
     * Computes the numerical inter-cell flux as
     *   F = ½(F_L + F_R) − ½ |A_Roe| (U_R − U_L)
     * where |A_Roe| is the Roe matrix with eigenvalues modified by the chosen
     * entropy fix (see Roe_EntropyFixer()).
     *
     * When @p eigScheme = 2 (Lax-Friedrichs), the function takes an early-exit
     * path returning the vanilla Lax-Friedrichs flux without Roe decomposition.
     *
     * @tparam dim        Spatial dimension (2 or 3).
     * @tparam eigScheme  Compile-time entropy-fix selector (0–8); default 0
     *                    (Harten-Yee).
     * @param  UL, UR     Left/right conservative face states.
     * @param  ULm, URm   Left/right mean states for Roe averaging.
     * @param  vg          Grid velocity vector.
     * @param  n           Face-normal vector.
     * @param  gamma       Ratio of specific heats (γ).
     * @param[out] F       Numerical flux (first dim+2 entries written).
     * @param  dLambda     H-correction transverse eigenvalue estimate.
     * @param  fixScale    Entropy-fix scaling factor (settings.rsFixScale).
     * @param  incFScale   Incremental flux dissipation scale (settings.rsIncFScale).
     * @param  dumpInfo    Debug dump callable for assertion failures.
     * @param[out] lam0    |V_n − a| after entropy fixing.
     * @param[out] lam123  |V_n| after entropy fixing.
     * @param[out] lam4    |V_n + a| after entropy fixing.
     */
    template <int dim = 3, int eigScheme = 0,
              typename TUL, typename TUR,
              typename TULm, typename TURm,
              typename TVecVG, typename TVecN,
              typename TF, typename TFdumpInfo>
    void RoeFlux_IdealGas_HartenYee(const TUL &UL, const TUR &UR,
                                    const TULm &ULm, const TURm &URm,
                                    const TVecVG &vg, const TVecN &n,
                                    real gamma, TF &F, real dLambda, real fixScale,
                                    real incFScale,
                                    const TFdumpInfo &dumpInfo, real &lam0, real &lam123, real &lam4)
    {
        using TVec = Eigen::Vector<real, dim>;

        if (!(UL(0) > 0 && UR(0) > 0))
        {
            dumpInfo();
        }
        DNDS_assert(UL(0) > 0 && UR(0) > 0);
        TVec veloL = (UL(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / UL(0)).matrix();
        TVec veloR = (UR(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / UR(0)).matrix();

        real asqrL, asqrR, pL, pR, HL, HR;
        real vsqrL = veloL.squaredNorm();
        real vsqrR = veloR.squaredNorm();
        IdealGasThermal(UL(dim + 1), UL(0), vsqrL, gamma, pL, asqrL, HL);
        IdealGasThermal(UR(dim + 1), UR(0), vsqrR, gamma, pR, asqrR, HR);

        auto rp = ComputeRoePreamble<dim>(ULm, URm, gamma, dumpInfo);

        Eigen::Vector<real, dim + 2> FL, FR;
        GasInviscidFlux_XY<dim>(UL, veloL, vg, n, pL, FL);
        GasInviscidFlux_XY<dim>(UR, veloR, vg, n, pR, FR);

        real veloRoeN = rp.veloRoe.dot(n);
        real vgN = vg.dot(n);
        real veloRoe0 = veloRoeN - vgN;
        lam0 = std::abs(veloRoe0 - rp.aRoe);
        lam123 = std::abs(veloRoe0);
        lam4 = std::abs(veloRoe0 + rp.aRoe);
        real veloLm0 = (rp.veloLm - vg).dot(n);
        real veloRm0 = (rp.veloRm - vg).dot(n);

        if constexpr (eigScheme == 2)
        {
            // *vanilla Lax
            // lam0 = lam123 = lam4 = std::max({lam0, lam123, lam4});
            lam0 = lam123 = lam4 = std::max(std::abs(veloLm0) + std::sqrt(rp.asqrLm), std::abs(veloRm0) + std::sqrt(rp.asqrRm));
            F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) =
                (FL + FR) * 0.5 -
                0.5 * lam0 * (UR(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) - UL(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)));
            return; //* early exit
        }
        else
            Roe_EntropyFixer<eigScheme>(std::sqrt(rp.asqrLm), std::sqrt(rp.asqrRm), rp.aRoe,
                                        veloLm0, veloRm0, veloRoe0,
                                        (rp.veloLm - vg).norm(), (rp.veloRm - vg).norm(), (rp.veloRoe - vg).norm(),
                                        dLambda, fixScale, incFScale,
                                        lam0, lam123, lam4);

        Eigen::Vector<real, dim + 2> incU =
            UR(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) -
            UL(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)); //! not using m, for this is accuracy-limited!
        real incU123N = incU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).dot(n);

        TVec alpha23V = incU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) - incU(0) * rp.veloRoe;
        TVec alpha23VT = alpha23V - n * alpha23V.dot(n);
        real incU4b = incU(dim + 1) - alpha23VT.dot(rp.veloRoe);
        real alpha1 = (gamma - 1) / rp.asqrRoe *
                      (incU(0) * (rp.HRoe - veloRoeN * veloRoeN) +
                       veloRoeN * incU123N - incU4b);
        real alpha0 = (incU(0) * (veloRoeN + rp.aRoe) - incU123N - rp.aRoe * alpha1) / (2 * rp.aRoe);
        real alpha4 = incU(0) - (alpha0 + alpha1);

        alpha0 *= lam0;
        alpha1 *= lam123;
        alpha23VT *= lam123;
        alpha4 *= lam4; // here becomes alpha_i * lam_i

        Eigen::Vector<real, dim + 2> incF;
        incF(0) = alpha0 + alpha1 + alpha4;
        incF(dim + 1) = (rp.HRoe - veloRoeN * rp.aRoe) * alpha0 + 0.5 * rp.vsqrRoe * alpha1 +
                        (rp.HRoe + veloRoeN * rp.aRoe) * alpha4 + alpha23VT.dot(rp.veloRoe);
        incF(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) =
            (rp.veloRoe - rp.aRoe * n) * alpha0 + (rp.veloRoe + rp.aRoe * n) * alpha4 +
            rp.veloRoe * alpha1 + alpha23VT;

        F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>)) = (FL + FR) * 0.5 - 0.5 * incF;
    }

    /**
     * @brief Computes the Roe-averaged state vector including passive scalars
     *        (e.g. RANS turbulence variables).
     *
     * The Roe average for the Euler components (density, momentum, energy) is
     * the standard √ρ-weighted mean. Passive scalars beyond index dim+1 are
     * also averaged using the same √ρ weighting.
     *
     * @tparam dim  Spatial dimension (2 or 3).
     * @param  UL, UR   Left/right conservative state vectors (full length
     *                   including any passive scalars).
     * @param  gamma     Ratio of specific heats (γ).
     * @param[out] veloRoe   Roe-averaged velocity vector.
     * @param[out] vsqrRoe   Roe-averaged squared velocity magnitude.
     * @param[out] aRoe      Roe-averaged speed of sound.
     * @param[out] asqrRoe   Roe-averaged speed of sound squared.
     * @param[out] HRoe      Roe-averaged total specific enthalpy.
     * @param[out] UOut      Roe-averaged state vector (same layout as UL/UR,
     *                       including passive scalars).
     */
    template <int dim = 3, typename TUL, typename TUR, typename TVecV, typename TUOut>
    void GetRoeAverage(const TUL &UL, const TUR &UR, real gamma,
                       TVecV &veloRoe, real &vsqrRoe, real &aRoe, real &asqrRoe, real &HRoe, TUOut &UOut)
    {
        using TVec = Eigen::Vector<real, dim>;
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto I4 = dim + 1;
        TVec veloLm = (UL(Seq123).array() / UL(0)).matrix();
        TVec veloRm = (UR(Seq123).array() / UR(0)).matrix();
        real asqrLm, asqrRm, pLm, pRm, HLm, HRm;
        real vsqrLm = veloLm.squaredNorm();
        real vsqrRm = veloRm.squaredNorm();
        IdealGasThermal(UL(dim + 1), UL(0), vsqrLm, gamma, pLm, asqrLm, HLm);
        IdealGasThermal(UR(dim + 1), UR(0), vsqrRm, gamma, pRm, asqrRm, HRm);
        DNDS_assert(UL(0) >= 0 && UR(0) >= 0);
        real sqrtRhoLm = std::sqrt(UL(0));
        real sqrtRhoRm = std::sqrt(UR(0));

        veloRoe = (sqrtRhoLm * veloLm + sqrtRhoRm * veloRm) / (sqrtRhoLm + sqrtRhoRm);
        vsqrRoe = veloRoe.squaredNorm();
        HRoe = (sqrtRhoLm * HLm + sqrtRhoRm * HRm) / (sqrtRhoLm + sqrtRhoRm);
        asqrRoe = (gamma - 1) * (HRoe - 0.5 * vsqrRoe);
        DNDS_assert(asqrRoe >= 0);
        aRoe = std::sqrt(asqrRoe);

        UOut(0) = sqrtRhoLm * sqrtRhoRm;
        UOut(Seq123) = veloRoe * UOut(0);
        real pRoe = asqrRoe * UOut(0) / gamma;
        UOut(I4) = pRoe / (gamma - 1) + 0.5 * vsqrRoe * UOut(0);
        UOut(Eigen::seq(I4 + 1, EigenLast)) =
            UOut(0) / (sqrtRhoLm + sqrtRhoRm) *
            (UL(Eigen::seq(I4 + 1, EigenLast)) / sqrtRhoLm +
             UR(Eigen::seq(I4 + 1, EigenLast)) / sqrtRhoRm);
    }

    /**
     * @brief Computes the dissipation part of the Roe flux |A|·dU, given
     *        pre-computed Roe averages and entropy-fixed eigenvalues.
     *
     * This function **accumulates** into @p incF (does not zero it first).
     * It handles the Euler components (indices 0..dim+1) via the standard Roe
     * eigenvector decomposition, and passive-scalar components (indices
     * dim+2..end) with the contact/shear eigenvalue lam123.
     *
     * Used by the LUSGS implicit solver where the dissipation term is needed
     * separately from the central flux.
     *
     * @tparam dim  Spatial dimension (2 or 3).
     * @param  incU     State jump U_R − U_L (full length, including scalars).
     * @param  n        Face-normal vector.
     * @param  veloRoe  Roe-averaged velocity.
     * @param  vsqrRoe  Roe-averaged |v|².
     * @param  aRoe     Roe-averaged speed of sound.
     * @param  asqrRoe  Roe-averaged a².
     * @param  HRoe     Roe-averaged total enthalpy.
     * @param  lam0     Entropy-fixed eigenvalue |V_n − a|.
     * @param  lam123   Entropy-fixed eigenvalue |V_n|.
     * @param  lam4     Entropy-fixed eigenvalue |V_n + a|.
     * @param  gamma    Ratio of specific heats (γ).
     * @param[in,out] incF  Dissipation flux; accumulated (not zeroed).
     */
    template <int dim = 3, typename TDU, typename TDF, typename TVecV, typename TVecN>
    void RoeFluxIncFDiff(const TDU &incU, const TVecN &n, const TVecV &veloRoe,
                         real vsqrRoe, real aRoe, real asqrRoe, real HRoe,
                         real lam0, real lam123, real lam4, real gamma,
                         TDF &incF)
    {
        static const auto SeqI52Last = Eigen::seq(Eigen::fix<dim + 2>, EigenLast);
        using TVec = Eigen::Vector<real, dim>;
        real veloRoeN = veloRoe.dot(n);

        real incU123N = incU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).dot(n);

        TVec alpha23V = incU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) - incU(0) * veloRoe;
        TVec alpha23VT = alpha23V - n * alpha23V.dot(n);
        real incU4b = incU(dim + 1) - alpha23VT.dot(veloRoe);
        real alpha1 = (gamma - 1) / asqrRoe *
                      (incU(0) * (HRoe - veloRoeN * veloRoeN) +
                       veloRoeN * incU123N - incU4b);
        real alpha0 = (incU(0) * (veloRoeN + aRoe) - incU123N - aRoe * alpha1) / (2 * aRoe);
        real alpha4 = incU(0) - (alpha0 + alpha1);

        alpha0 *= lam0;
        alpha1 *= lam123;
        alpha23VT *= lam123;
        alpha4 *= lam4; // here becomes alpha_i * lam_i

        incF(0) += alpha0 + alpha1 + alpha4;
        incF(dim + 1) += (HRoe - veloRoeN * aRoe) * alpha0 + 0.5 * vsqrRoe * alpha1 +
                         (HRoe + veloRoeN * aRoe) * alpha4 + alpha23VT.dot(veloRoe);
        incF(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)) +=
            (veloRoe - aRoe * n) * alpha0 + (veloRoe + aRoe * n) * alpha4 +
            veloRoe * alpha1 + alpha23VT;
        // ! handling the passive part of U!
        incF(SeqI52Last) += incU(SeqI52Last) * lam123;
    }

    /**
     * @brief Batched Roe flux with selectable entropy fix for column-major
     *        state matrices.
     *
     * Evaluates the Roe flux for @c nB = UL.cols() faces simultaneously. The
     * Roe averaging is performed from the single-column mean states @p ULm,
     * @p URm, sharing one set of Roe eigenvalues across the batch. Per-face
     * normal vectors @p n and grid velocities @p vg allow each column to have
     * a distinct orientation and motion; @p nm and @p vgm are the mean-state
     * normal/grid velocity used for the shared Roe average.
     *
     * @tparam dim        Spatial dimension (2 or 3).
     * @tparam eigScheme  Compile-time entropy-fix selector (0–8); default 0.
     * @param  UL, UR     Left/right face states [(dim+2) × nBatch].
     * @param  ULm, URm   Mean-state vectors for Roe averaging (single column).
     * @param  vg, vgm    Per-face and mean-state grid velocities.
     * @param  n, nm       Per-face and mean-state face normals.
     * @param  gamma       Ratio of specific heats (γ).
     * @param[out] F       Numerical flux matrix [(dim+2) × nBatch].
     * @param  dLambda     H-correction transverse eigenvalue estimate.
     * @param  fixScale    Entropy-fix scaling factor.
     * @param  incFScale   Incremental flux dissipation scale.
     * @param  dumpInfo    Debug dump callable.
     * @param[out] lam0    |V_n − a| after entropy fixing.
     * @param[out] lam123  |V_n| after entropy fixing.
     * @param[out] lam4    |V_n + a| after entropy fixing.
     */
    template <int dim = 3, int eigScheme = 0,
              typename TUL, typename TUR,
              typename TULm, typename TURm,
              typename TVecVG, typename TVecVGm,
              typename TVecN, typename TVecNm,
              typename TF, typename TFdumpInfo>
    void RoeFlux_IdealGas_HartenYee_Batch(const TUL &UL, const TUR &UR,
                                          const TULm &ULm, const TURm &URm,
                                          const TVecVG &vg, const TVecVGm &vgm,
                                          const TVecN &n, const TVecNm &nm,
                                          real gamma, TF &F,
                                          real dLambda, real fixScale,
                                          real incFScale,
                                          const TFdumpInfo &dumpInfo, real &lam0, real &lam123, real &lam4)
    {
        using TVec = Eigen::Vector<real, dim>;
        using TVec_Batch = Eigen::Matrix<real, dim, -1, Eigen::ColMajor, dim, MaxBatch>;
        using TReal_Batch = Eigen::Matrix<real, 1, -1, Eigen::RowMajor, 1, MaxBatch>;
        using TU5_Batch = Eigen::Matrix<real, dim + 2, -1, Eigen::ColMajor, dim + 2, MaxBatch>;

        int nB = UL.cols();
        for (int iB = 0; iB < nB; iB++)
            if (!(UL(0, iB) > 0 && UR(0, iB) > 0))
            {
                dumpInfo();
                DNDS_assert(false);
            }
        TVec_Batch veloL = (UL(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>), EigenAll).array().rowwise() / UL(0, EigenAll).array()).matrix();
        TVec_Batch veloR = (UR(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>), EigenAll).array().rowwise() / UR(0, EigenAll).array()).matrix();
        TVec veloLm = (ULm(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / ULm(0)).matrix();
        TVec veloRm = (URm(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).array() / URm(0)).matrix();

        TReal_Batch pL, pR;
        pL.resize(nB), pR.resize(nB);
        for (int iB = 0; iB < nB; iB++)
        {
            real asqrL, asqrR, HL, HR;
            real vsqrL = veloL(EigenAll, iB).squaredNorm();
            real vsqrR = veloR(EigenAll, iB).squaredNorm();
            IdealGasThermal(UL(dim + 1, iB), UL(0, iB), vsqrL, gamma, pL(iB), asqrL, HL);
            IdealGasThermal(UR(dim + 1, iB), UR(0, iB), vsqrR, gamma, pR(iB), asqrR, HR);
        }

        real asqrLm, asqrRm, pLm, pRm, HLm, HRm;
        real vsqrLm = veloLm.squaredNorm();
        real vsqrRm = veloRm.squaredNorm();
        IdealGasThermal(ULm(dim + 1), ULm(0), vsqrLm, gamma, pLm, asqrLm, HLm);
        IdealGasThermal(URm(dim + 1), URm(0), vsqrRm, gamma, pRm, asqrRm, HRm);

        real sqrtRhoLm = std::sqrt(ULm(0));
        real sqrtRhoRm = std::sqrt(URm(0));

        TVec veloRoe = (sqrtRhoLm * veloLm + sqrtRhoRm * veloRm) / (sqrtRhoLm + sqrtRhoRm);
        real vsqrRoe = veloRoe.squaredNorm();
        real HRoe = (sqrtRhoLm * HLm + sqrtRhoRm * HRm) / (sqrtRhoLm + sqrtRhoRm);
        real asqrRoe = (gamma - 1) * (HRoe - 0.5 * vsqrRoe);
        real rhoRoe = sqrtRhoLm * sqrtRhoRm;
        real veloRoeN = veloRoe.dot(nm);
        real vgmN = vgm.dot(nm);
        real veloRoeRN = veloRoeN - vgmN;
        real veloLm0 = (veloLm - vgm).dot(nm);
        real veloRm0 = (veloRm - vgm).dot(nm);

        TU5_Batch FL, FR;
        FL.resize(Eigen::NoChange, UL.cols());
        FR.resize(Eigen::NoChange, UL.cols());
        GasInviscidFlux_XY_Batch<dim>(UL, veloL, vg, n, pL, FL);
        GasInviscidFlux_XY_Batch<dim>(UR, veloR, vg, n, pR, FR);

        if (!(asqrRoe > 0))
        {
            dumpInfo();
        }
        DNDS_assert(asqrRoe > 0);
        real aRoe = std::sqrt(asqrRoe);

        lam0 = std::abs(veloRoeRN - aRoe);
        lam123 = std::abs(veloRoeRN);
        lam4 = std::abs(veloRoeRN + aRoe);

        if constexpr (eigScheme == 2)
        {
            // *vanilla Lax
            // lam0 = lam123 = lam4 = std::max({lam0, lam123, lam4});
            lam0 = lam123 = lam4 = std::max(std::abs(veloLm0) + std::sqrt(asqrLm), std::abs(veloRm0) + std::sqrt(asqrRm));
            F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>), EigenAll) =
                (FL + FR) * 0.5 -
                0.5 * lam0 * (UR(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>), EigenAll) - UL(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>), EigenAll));
            return; //* early exit
        }
        else
            Roe_EntropyFixer<eigScheme>(std::sqrt(asqrLm), std::sqrt(asqrRm), aRoe,
                                        veloLm0, veloRm0, veloRoeRN,
                                        (veloLm - vgm).norm(), (veloRm - vgm).norm(), (veloRoe - vgm).norm(),
                                        dLambda, fixScale, incFScale,
                                        lam0, lam123, lam4);

        TU5_Batch incU =
            UR(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>), EigenAll) -
            UL(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>), EigenAll); //! not using m, for this is accuracy-limited!
        TReal_Batch incU123N =
            (incU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>), EigenAll).array() * n.array()).colwise().sum();
        TVec_Batch alpha23V = incU(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>), EigenAll) - veloRoe * incU(0, EigenAll);
        TVec_Batch alpha23VT = alpha23V.array() - n.array().rowwise() * (alpha23V.array() * n.array()).colwise().sum();
        TReal_Batch incU4b =
            incU(dim + 1, EigenAll) -
            veloRoe.transpose() * alpha23VT;
        TReal_Batch alpha1 =
            (gamma - 1) / asqrRoe *
            (incU(0, EigenAll) * (HRoe - veloRoeN * veloRoeN) +
             veloRoeN * incU123N - incU4b);
        TReal_Batch alpha0 =
            (incU(0, EigenAll) * (veloRoeN + aRoe) - incU123N - aRoe * alpha1) / (2 * aRoe);
        TReal_Batch alpha4 =
            incU(0, EigenAll) - (alpha0 + alpha1);

        alpha0 *= lam0;
        alpha1 *= lam123;
        alpha23VT *= lam123;
        alpha4 *= lam4; // here becomes alpha_i * lam_i

        TU5_Batch incF;
        incF.resize(Eigen::NoChange, UL.cols());
        incF(0, EigenAll) = alpha0 + alpha1 + alpha4;
        incF(dim + 1, EigenAll) = (HRoe - veloRoeN * aRoe) * alpha0 + 0.5 * vsqrRoe * alpha1 +
                                  (HRoe + veloRoeN * aRoe) * alpha4 + veloRoe.transpose() * alpha23VT;
        incF(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>), EigenAll) =
            ((-aRoe * n).array().colwise() + veloRoe.array()).rowwise() * alpha0.array() +
            ((aRoe * n).array().colwise() + veloRoe.array()).rowwise() * alpha4.array() +
            alpha23VT.array() +
            (veloRoe * alpha1).array();
        // incF(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>), EigenAll) =
        //     (veloRoe.array() - (aRoe * n).array().colwise()) * alpha0 * (veloRoe.array() + (aRoe * n).array().colwise()) * alpha4;

        F(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>), EigenAll) = (FL + FR) * 0.5 - 0.5 * incF;
    }

    /**
     * @brief Runtime dispatcher from RiemannSolverType to the correct Roe /
     *        HLLC / HLLEP template instantiation.
     *
     * Maps RiemannSolverType enumerators to the corresponding compile-time
     * eigScheme / type template arguments. For Roe variants, calls
     * RoeFlux_IdealGas_HartenYee; for HLLEP, calls HLLEPFlux_IdealGas; for
     * HLLC, calls HLLCFlux_IdealGas_HartenYee.
     *
     * @tparam dim  Spatial dimension (2 or 3).
     * @param  type       Riemann solver selector (runtime).
     * @param  UL, UR     Left/right conservative face states.
     * @param  ULm, URm   Mean states for Roe averaging.
     * @param  vg          Grid velocity.
     * @param  n           Face-normal.
     * @param  gamma       Ratio of specific heats (γ).
     * @param[out] F       Numerical flux.
     * @param  dLambda     H-correction transverse eigenvalue estimate.
     * @param  fixScale    Entropy-fix scaling factor.
     * @param  incFScale   Incremental flux dissipation scale.
     * @param  dumpInfo    Debug dump callable.
     * @param[out] lam0    |V_n − a| after entropy fixing.
     * @param[out] lam123  |V_n| after entropy fixing.
     * @param[out] lam4    |V_n + a| after entropy fixing.
     */
    template <int dim = 3,
              typename TUL, typename TUR,
              typename TULm, typename TURm,
              typename TVecVG, typename TVecN,
              typename TF, typename TFdumpInfo>
    void InviscidFlux_IdealGas_Dispatcher(
        RiemannSolverType type,
        TUL &&UL, TUR &&UR, TULm &&ULm, TURm &&URm,
        TVecVG &&vg, TVecN &&n, real gamma, TF &&F,
        real dLambda, real fixScale,
        real incFScale,
        TFdumpInfo &&dumpInfo, real &lam0, real &lam123, real &lam4)
    {
#define DNDS_GAS_CALL_ROE(type)                                          \
    RoeFlux_IdealGas_HartenYee<dim, type>(                               \
        UL, UR, ULm, URm, vg, n, gamma, F, dLambda, fixScale, incFScale, \
        dumpInfo, lam0, lam123, lam4)

        if (type == Roe)
            RoeFlux_IdealGas_HartenYee<dim>(
                UL, UR, ULm, URm, vg, n, gamma, F, dLambda, fixScale, incFScale,
                dumpInfo, lam0, lam123, lam4);
        else if (type == Roe_M1)
            DNDS_GAS_CALL_ROE(1);
        else if (type == Roe_M2)
            DNDS_GAS_CALL_ROE(2);
        else if (type == Roe_M3)
            DNDS_GAS_CALL_ROE(3);
        else if (type == Roe_M4)
            DNDS_GAS_CALL_ROE(4);
        else if (type == Roe_M5)
            DNDS_GAS_CALL_ROE(5);
        else if (type == Roe_M6)
            DNDS_GAS_CALL_ROE(6);
        else if (type == Roe_M7)
            DNDS_GAS_CALL_ROE(7);
        else if (type == Roe_M8)
            DNDS_GAS_CALL_ROE(8);
        else if (type == Roe_M9)
            DNDS_GAS_CALL_ROE(9);
        else if (type == HLLEP)
            HLLEPFlux_IdealGas<dim, 0>(
                UL, UR, ULm, URm, vg, n, gamma, F, dLambda, fixScale,
                dumpInfo, lam0, lam123, lam4);
        else if (type == HLLEP_V1)
            HLLEPFlux_IdealGas<dim, 1>(
                UL, UR, ULm, URm, vg, n, gamma, F, dLambda, fixScale,
                dumpInfo, lam0, lam123, lam4);
        else if (type == HLLC)
            HLLCFlux_IdealGas_HartenYee<dim>(
                UL, UR, ULm, URm, vg, n, gamma, F, dLambda, fixScale,
                dumpInfo, lam0, lam123, lam4);
        else
            DNDS_assert_info(false, "the rs type is invalid");
#undef DNDS_GAS_CALL_ROE
    }

    /**
     * @brief Runtime dispatcher for batched Roe flux from RiemannSolverType to
     *        the correct RoeFlux_IdealGas_HartenYee_Batch template instantiation.
     *
     * Only Roe variants (Roe, Roe_M1..M9) are supported in batch mode; HLLC
     * and HLLEP are not available and will trigger an assertion failure.
     *
     * @tparam dim  Spatial dimension (2 or 3).
     * @param  type       Riemann solver selector (runtime).
     * @param  UL, UR     Left/right face states [(dim+2) × nBatch].
     * @param  ULm, URm   Mean states for Roe averaging (single column).
     * @param  vg, vgm    Per-face and mean-state grid velocities.
     * @param  n, nm       Per-face and mean-state face normals.
     * @param  gamma       Ratio of specific heats (γ).
     * @param[out] F       Numerical flux matrix [(dim+2) × nBatch].
     * @param  dLambda     H-correction transverse eigenvalue estimate.
     * @param  fixScale    Entropy-fix scaling factor.
     * @param  incFScale   Incremental flux dissipation scale.
     * @param  dumpInfo    Debug dump callable.
     * @param[out] lam0    |V_n − a| after entropy fixing.
     * @param[out] lam123  |V_n| after entropy fixing.
     * @param[out] lam4    |V_n + a| after entropy fixing.
     */
    template <int dim = 3,
              typename TUL, typename TUR,
              typename TULm, typename TURm,
              typename TVecVG, typename TVecVGm,
              typename TVecN, typename TVecNm,
              typename TF, typename TFdumpInfo>
    void InviscidFlux_IdealGas_Batch_Dispatcher(
        RiemannSolverType type,
        TUL &&UL, TUR &&UR,
        TULm &&ULm, TURm &&URm,
        TVecVG &&vg, TVecVGm &&vgm,
        TVecN &&n, TVecNm &&nm,
        real gamma, TF &&F, real dLambda, real fixScale,
        real incFScale,
        TFdumpInfo &dumpInfo, real &lam0, real &lam123, real &lam4)
    {
#define DNDS_GAS_CALL_ROE(type)                                                   \
    RoeFlux_IdealGas_HartenYee_Batch<dim, type>(                                  \
        UL, UR, ULm, URm, vg, vgm, n, nm, gamma, F, dLambda, fixScale, incFScale, \
        dumpInfo, lam0, lam123, lam4)

        if (type == Roe)
            RoeFlux_IdealGas_HartenYee_Batch<dim>(
                UL, UR, ULm, URm, vg, vgm, n, nm, gamma, F, dLambda, fixScale, incFScale,
                dumpInfo, lam0, lam123, lam4);
        else if (type == Roe_M1)
            DNDS_GAS_CALL_ROE(1);
        else if (type == Roe_M2)
            DNDS_GAS_CALL_ROE(2);
        else if (type == Roe_M3)
            DNDS_GAS_CALL_ROE(3);
        else if (type == Roe_M4)
            DNDS_GAS_CALL_ROE(4);
        else if (type == Roe_M5)
            DNDS_GAS_CALL_ROE(5);
        else if (type == Roe_M6)
            DNDS_GAS_CALL_ROE(6);
        else if (type == Roe_M7)
            DNDS_GAS_CALL_ROE(7);
        else if (type == Roe_M8)
            DNDS_GAS_CALL_ROE(8);
        else if (type == Roe_M9)
            DNDS_GAS_CALL_ROE(9);
        else
            DNDS_assert_info(false, "the rs type is invalid (for batch version)");
#undef DNDS_GAS_CALL_ROE
    }

    /**
     * @brief Computes the viscous (Navier-Stokes) flux projected onto a face
     *        normal for an ideal gas.
     *
     * Evaluates the viscous stress tensor τ (including Stokes' hypothesis
     * λ = −2/3 μ) and the heat flux vector q = −k ∇T, then projects onto
     * @p norm.  Supports:
     * - Sutherland-law viscosity (via the externally provided @p mu).
     * - Turbulent eddy viscosity through @p mutRatio (μ_t / μ).
     * - QCR (Quadratic Constitutive Relation) correction when @p mutQCRFix
     *   is true, using ccr1 = 0.3.
     * - Adiabatic wall treatment: when @p adiabatic is true, the wall-normal
     *   component of ∇T is removed.
     *
     * @note @p GradUPrim has size [dim × nVars]; it is the gradient of the
     *       **primitive** variables (ρ, u, v, [w,] p, ...) in physical space.
     *       "3×5 TGradU" in the original comment refers to dim=3 with 5 Euler
     *       variables.
     *
     * @tparam dim  Spatial dimension (2 or 3).
     * @param  U          Conservative state vector.
     * @param  GradUPrim  Gradient of primitive variables [dim × nVars].
     * @param  norm       Face-normal vector (unit or area-weighted).
     * @param  adiabatic  If true, zero the wall-normal heat flux component.
     * @param  gamma      Ratio of specific heats (γ).
     * @param  mu         Dynamic (molecular + turbulent) viscosity μ + μ_t.
     * @param  mutRatio   Turbulent-to-molecular viscosity ratio μ_t / μ.
     * @param  mutQCRFix  Enable QCR stress correction.
     * @param  k          Thermal conductivity (molecular + turbulent).
     * @param  Cp         Specific heat at constant pressure.
     * @param[out] Flux   Viscous flux vector (dim+2 entries, Flux[0] = 0).
     */
    template <int dim = 3, typename TU, typename TGradU, typename TFlux, typename TNorm>
    void ViscousFlux_IdealGas(const TU &U, const TGradU &GradUPrim, TNorm norm, bool adiabatic, real gamma,
                              real mu, real mutRatio, bool mutQCRFix, real k, real Cp, TFlux &Flux)
    {
        static const auto Seq01234 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);

        Eigen::Vector<real, dim> velo = U(Seq123) / U(0);
        static const real lambda = -2. / 3.;
        Eigen::Matrix<real, dim, dim> diffVelo = GradUPrim(Seq012, Seq123); // dU_j/dx_i
        Eigen::Vector<real, dim> GradP = GradUPrim(Seq012, dim + 1);
        real vSqr = velo.squaredNorm();
        real p = (gamma - 1) * (U(dim + 1) - U(0) * 0.5 * vSqr);
        Eigen::Vector<real, dim> GradT = (gamma / ((gamma - 1) * Cp * U(0) * U(0))) *
                                         (U(0) * GradP - p * GradUPrim(Seq012, 0)); // GradU(:,0) is grad rho no matter prim or not

        if (adiabatic) //! is this fix reasonable?
            GradT -= GradT.dot(norm) * norm;

        Eigen::Matrix<real, dim, dim> vStress = (diffVelo + diffVelo.transpose()) * mu +
                                                Eigen::Matrix<real, dim, dim>::Identity() * (lambda * mu * diffVelo.trace());
        if (mutQCRFix)
        {
            real b = std::sqrt((diffVelo.array() * diffVelo.array()).sum());
            Eigen::Matrix<real, dim, dim> O = (diffVelo.transpose() - diffVelo) / (b + verySmallReal); // dU_i/dx_j-dU_j/dx_i
            real ccr1 = 0.3;
            Eigen::Matrix<real, dim, dim> vStressQCRFix;
            vStressQCRFix.setZero();
            vStressQCRFix.diagonal() = (vStress.array() * O.array()).rowwise().sum();
            vStressQCRFix(0, 1) = O(0, 1) * (vStress(1, 1) - vStress(0, 0));
            if (dim == 3)
            {
                vStressQCRFix(0, 1) += O(1, 2) * vStress(0, 2) + O(0, 2) * vStress(1, 2);
                vStressQCRFix(0, 2) = O(0, 2) * (vStress(2, 2) - vStress(0, 0)) + O(0, 1) * vStress(2, 1) + O(2, 1) * vStress(0, 1);
                vStressQCRFix(1, 2) = O(1, 2) * (vStress(2, 2) - vStress(1, 1)) + O(1, 0) * vStress(2, 0) + O(2, 0) * vStress(1, 0);
            }
            Eigen::Matrix<real, dim, dim> vStressQCRFixFull = vStressQCRFix + vStressQCRFix.transpose();
            vStress -= ccr1 * mutRatio * vStressQCRFixFull;
        }
        Flux(0) = 0;
        Flux(Seq123) = vStress * norm;
        Flux(dim + 1) = (vStress * velo + k * GradT).dot(norm);
        if (!Flux.allFinite())
        {
            std::cout << "U\n"
                      << U.transpose() << "\n";
            std::cout << "GradUPrim\n"
                      << GradUPrim << "\n";
            std::cout << "norm\n"
                      << norm << "\n";
            std::cout << "gamma\n"
                      << gamma << "\n";
            std::cout << "mu\n"
                      << mu << "\n";
            std::cout << "k\n"
                      << k << "\n";
            std::cout << "Cp\n"
                      << Cp << "\n";
            DNDS_assert(false);
        }
    }

    /**
     * @brief Converts the gradient of conservative variables to the gradient of
     *        primitive variables for an ideal gas.
     *
     * Given GradU = ∂U/∂x (the spatial gradient of the conservative vector),
     * computes GradUPrim = ∂(ρ,u,v,[w,]p,...)/∂x in-place. The density
     * gradient column (column 0) is unchanged, velocity gradient columns are
     * transformed via the quotient rule, and the pressure gradient column uses
     * the chain rule through the equation of state.
     *
     * @note The input @p GradU has size [dim × nVars], where nVars ≥ dim+2.
     *       Passive-scalar gradient columns beyond index dim+1 are also
     *       converted (divided by ρ after removing the density contribution).
     *
     * @tparam dim  Spatial dimension (2 or 3).
     * @param  U            Conservative state vector.
     * @param  GradU        Gradient of conservative variables [dim × nVars].
     * @param[out] GradUPrim Gradient of primitive variables [dim × nVars].
     * @param  gamma        Ratio of specific heats (γ).
     */
    template <int dim = 3, typename TU, typename TGradU, typename TGradUPrim>
    void GradientCons2Prim_IdealGas(const TU &U, const TGradU &GradU, TGradUPrim &GradUPrim, real gamma)
    {
        static const auto Seq01234 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto I4 = dim + 1;

        Eigen::Vector<real, dim> velo = U(Seq123) / U(0);
        GradUPrim = GradU;

        GradUPrim(Seq012, Seq123) = (1.0 / sqr(U(0))) *
                                    (U(0) * GradU(Seq012, Seq123) -
                                     GradU(Seq012, 0) * Eigen::RowVector<real, dim>(U(Seq123))); // dU_j/dx_i
        GradUPrim(Seq012, I4) = (gamma - 1) *
                                (GradU(Seq012, dim + 1) -
                                 0.5 *
                                     (GradU(Seq012, Seq123) * velo +
                                      GradUPrim(Seq012, Seq123) * Eigen::Vector<real, dim>(U(Seq123))));
        GradUPrim(Seq012, Eigen::seq(Eigen::fix<I4 + 1>, EigenLast)) -= GradU(Seq012, 0) * U(Eigen::seq(Eigen::fix<I4 + 1>, EigenLast)).transpose() / U(0);
        GradUPrim(Seq012, Eigen::seq(Eigen::fix<I4 + 1>, EigenLast)) /= U(0);
    }

    /**
     * @brief Extracts the velocity gradient tensor from the conservative-variable
     *        gradient using the quotient rule.
     *
     * Returns diffVelo(i,j) = ∂u_j/∂x_i, a [dim × dim] matrix, computed as
     * (ρ · ∂(ρu_j)/∂x_i − (ρu_j) · ∂ρ/∂x_i) / ρ².
     *
     * @tparam dim  Spatial dimension (2 or 3).
     * @param  U      Conservative state vector.
     * @param  GradU  Gradient of conservative variables [dim × nVars].
     * @return Eigen::Matrix<real, dim, dim>  Velocity gradient tensor (dU_j/dx_i).
     */
    template <int dim, typename TU, typename TGradU>
    auto GetGradVelo(const TU &U, const TGradU &GradU)
    {
        static const auto Seq01234 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        Eigen::Matrix<real, dim, dim> diffVelo = (1.0 / sqr(U(0))) *
                                                 (U(0) * GradU(Seq012, Seq123) -
                                                  GradU(Seq012, 0) * Eigen::RowVector<real, dim>(U(Seq123))); // dU_j/dx_i
        return diffVelo;
    }

    /**
     * @brief Computes the maximum safe compression ratio (scaling factor α ∈ [0,1])
     *        for a conservative-variable increment that keeps internal energy
     *        above a prescribed positive floor.
     *
     * Given a state @p u and an increment @p uInc, finds the largest α such that
     * the internal energy e = E − ½ρ|v|² of (u + α·uInc) remains above
     * @p newrhoEinteralNew.  Two schemes are available:
     *   - scheme 0: analytic quadratic solve with iterative safety fallback.
     *   - scheme 1: convex (linear) estimation only.
     *
     * Used during implicit time stepping and limiting to prevent negative
     * pressures.
     *
     * TODO: vectorize
     *
     * @tparam dim         Spatial dimension (2 or 3).
     * @tparam scheme      0 = quadratic solve, 1 = convex estimate.
     * @tparam nVarsFixed  Compile-time size of the state vector.
     * @param  u           Current conservative state.
     * @param  uInc        Proposed conservative state increment.
     * @param  newrhoEinteralNew  Desired minimum internal energy floor ρ·e_min
     *                            (= p_min / (γ−1)).
     * @return The safe scaling factor α in [0, 1].
     */
    template <int dim = 3, int scheme = 0, int nVarsFixed, typename TU, typename TUInc>
    real IdealGasGetCompressionRatioPressure(const TU &u, const TUInc &uInc, real newrhoEinteralNew)
    {
        static const real safetyRatio = 1 - 1e-5;
        static const auto Seq01234 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto I4 = dim + 1;

        Eigen::Vector<real, nVarsFixed> ret = uInc;
        Eigen::Vector<real, nVarsFixed> uNew = u + uInc;
        real rhoEOld = u(I4) - u(Seq123).squaredNorm() / (u(0) + verySmallReal) * 0.5;
        newrhoEinteralNew = std::max(smallReal * rhoEOld, newrhoEinteralNew);
        real rhoENew = uNew(I4) - uNew(Seq123).squaredNorm() / (uNew(0) + verySmallReal) * 0.5;
        real alphaEst1 = (rhoEOld - newrhoEinteralNew) / std::max(-rhoENew + rhoEOld, verySmallReal);
        if (rhoENew > rhoEOld)
            alphaEst1 = 1;
        alphaEst1 = std::min(alphaEst1, 1.);
        alphaEst1 = std::max(alphaEst1, 0.);
        real alpha = alphaEst1; //! using convex estimation

        real alphaL, alphaR, c0, c1, c2;
        alphaL = alphaR = c0 = c1 = c2 = 0;
        if constexpr (scheme == 0)
        {
            c0 = 2 * u(I4) * u(0) - u(Seq123).squaredNorm() - 2 * u(0) * newrhoEinteralNew;
            c1 = 2 * u(I4) * ret(0) + 2 * u(0) * ret(I4) - 2 * u(Seq123).dot(ret(Seq123)) - 2 * ret(0) * newrhoEinteralNew;
            c2 = 2 * ret(I4) * ret(0) - ret(Seq123).squaredNorm();
            c2 += signP(c2) * verySmallReal;
            real deltaC = sqr(c1) - 4 * c0 * c2;
            if (deltaC <= -sqr(c0) * smallReal)
            {
                std::cout << std::scientific << std::setprecision(5);
                std::cout << u.transpose() << std::endl;
                std::cout << uInc.transpose() << std::endl;
                std::cout << newrhoEinteralNew << std::endl;
                std::cout << fmt::format("{} {} {}", c0, c1, c2) << std::endl;

                DNDS_assert(false);
            }
            deltaC = std::max(0., deltaC);
            real alphaL = (-std::sqrt(deltaC) - c1) / (2 * c2);
            real alphaR = (std::sqrt(deltaC) - c1) / (2 * c2);
            // if (c2 > 0)
            //     DNDS_assert(alphaL > 0);
            // DNDS_assert(alphaR > 0);
            // DNDS_assert(alphaL < 1);
            // if (c2 < 0)
            //     DNDS_assert(alphaR < 1);

            if (std::abs(c2) < 1e-10 * c0)
            {
                if (std::abs(c1) < 1e-10 * c0)
                {
                    alpha = 0;
                }
                else
                {
                    alpha = std::min(-c0 / c1, 1.);
                }
            }
            else
            {
                alpha = std::min((c2 > 0 ? alphaL : alphaL), 1.);
            }
            alpha = std::max(0., alpha);
            alpha *= safetyRatio;
            if (alpha < smallReal)
                alpha = 0;
        }
        else if constexpr (scheme == 1)
        {
            // has used convex
            alpha *= safetyRatio;
            if (alpha < smallReal)
                alpha = 0;
        }

        ret *= alpha;

        // Eigen::Vector<real, nVarsFixed> uNew = u + ret;
        // real eNew = uNew(I4) - 0.5 * uNew(Seq123).squaredNorm() / uNew(0);

        real decay = 1 - 1e-2;
        int iter;
        for (iter = 0; iter < 1000; iter++)
        {
            real ek = 0.5 * (u(Seq123) + ret(Seq123)).squaredNorm() / (u(0) + ret(0) + verySmallReal);
            if (ret(I4) + u(I4) - ek < newrhoEinteralNew)
            {

                ret *= decay, alpha *= decay;
            }
            else
                break;
        }
        if (iter >= 1000)
        {
            real ek = 0.5 * (u(Seq123) + ret(Seq123)).squaredNorm() / (u(0) + ret(0) + verySmallReal);
            {
                std::cout << std::scientific << std::setprecision(5);
                std::cout << fmt::format("alphas: {}, {}, {}\n", alpha, alphaL, alphaR);
                std::cout << fmt::format("ABC: {}, {}, {}\n", c2, c1, c0);
                std::cout << u.transpose() << std::endl;
                std::cout << uInc.transpose() << std::endl;
                std::cout << fmt::format("eks: {} {}\n", ret(I4) + u(I4) - ek, newrhoEinteralNew);
                DNDS_assert(false);
            }
        }
        // std::cout << fmt::format("{} {} {} {} {}", c0, c1, c2, alphaL, alphaR) << std::endl;

        return alpha;
    }
}