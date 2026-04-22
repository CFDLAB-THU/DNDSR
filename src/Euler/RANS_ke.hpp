/** @file RANS_ke.hpp
 *  @brief RANS two-equation turbulence model implementations for the DNDSR CFD solver.
 *
 *  Provides template functions for computing eddy viscosity, viscous fluxes, and
 *  source terms for several widely-used RANS turbulence closures:
 *
 *  - **Realizable k-epsilon** (Shih et al.) with f_mu damping and realizability constraint.
 *  - **k-omega SST** (Menter) with F1/F2 blending, cross-diffusion, and DES/DDES/IDDES support.
 *  - **k-omega Wilcox** (1988 / 2006 variants) with stress limiter and f_beta correction.
 *  - **Spalart-Allmaras** (SA) one-equation model with optional rotation correction,
 *    negative-nuTilde extension, and DES/DDES/IDDES/WMLES length-scale modification.
 *
 *  All functions are templated on spatial dimension @c dim and accept Eigen-compatible
 *  expression types for the conservative state vector @c UMeanXy and the gradient tensor
 *  @c DiffUxy (dim × nVars).  The convention for variable layout is:
 *  - Indices 0..dim: density and momentum components.
 *  - Index @c I4 = dim+1: total energy.
 *  - Index @c I4+1: first RANS variable (k or nuTilde).
 *  - Index @c I4+2: second RANS variable (epsilon or omega), where applicable.
 *
 *  Compile-time macros control model variants and limiters; see definitions below.
 *
 *  @note This file is included by other Euler module headers and should not normally
 *        be included directly by application code.
 */
#pragma once

#include "Euler.hpp"

/** @brief When set to 1, cap turbulent viscosity at 1e5 * mu_laminar for k-epsilon. */
#define KE_LIMIT_MUT 1

/** @brief Select Wilcox k-omega model version: 0 = original 1988, 1 = 2006 with stress limiter, 2 = simplified. */
#define KW_WILCOX_VER 1
/** @brief When set to 1, enable production limiters for Wilcox k-omega (CFL3D-style capping). */
#define KW_WILCOX_PROD_LIMITS 1
/** @brief When set to 1, cap turbulent viscosity at 1e5 * mu_laminar for Wilcox k-omega. */
#define KW_WILCOX_LIMIT_MUT 1

/** @brief When set to 1, cap turbulent viscosity at 1e5 * mu_laminar for SST. */
#define KW_SST_LIMIT_MUT 1
/** @brief When set to 1, enable production limiters for SST (CFL3D-style capping). */
#define KW_SST_PROD_LIMITS 1
/** @brief Select omega production formulation for SST: 0 = classical, 1 = strain-rate based, 2 = vorticity based. */
#define KW_SST_PROD_OMEGA_VERSION 1

/** @brief When set to 1, enable the ft2 laminar suppression term in Spalart-Allmaras. Setting to 0 disables ft2. */
#define SA_USE_FT2_TERM 1

/** @brief RANS turbulence model functions (eddy viscosity, viscous flux, source terms). */
namespace DNDS::Euler::RANS
{
    /** @brief Spalart-Allmaras model constants used across multiple files.
     *
     *  Canonical definitions from Spalart & Allmaras (1994).
     *  These constants are shared between the SA source term computation
     *  and boundary condition routines in other translation units.
     */
    namespace SA
    {
        static constexpr real cnu1 = 7.1;   ///< SA model constant c_{v1} controlling the viscous damping function f_{v1}.
        static constexpr real cn1 = 16.0;    ///< SA model constant c_{n1} for the negative-nuTilde extension function f_n.
        static constexpr real sigma = 2.0 / 3.0; ///< SA model diffusion coefficient sigma.
    }

    /** @brief Wall-omega boundary condition coefficient for k-omega family models.
     *
     *  Applied at wall boundaries as: rhoOmegaWall = mu / d^2 * kWallOmegaCoeff,
     *  where d is the wall-normal distance of the first cell center.
     */
    static constexpr real kWallOmegaCoeff = 800.0;

    /**
     * @brief Compute turbulent viscosity mu_t from the Shih et al. realizable k-epsilon model.
     *
     * Evaluates the eddy viscosity using the realizable k-epsilon formulation with:
     * - f_mu damping function based on the turbulent Reynolds number Re_t = rho * k^2 / (mu * epsilon).
     * - Realizability constraint: mu_t <= phi * rho * k / S, where S is the strain-rate magnitude.
     * - Optional upper bound of 1e5 * mu_laminar when @c KE_LIMIT_MUT is enabled.
     *
     * @tparam dim     Spatial dimension (2 or 3).
     * @tparam TU      Eigen-compatible type for the conservative state vector.
     * @tparam TDiffU  Eigen-compatible type for the gradient tensor (dim × nVars, conservative variables).
     * @param UMeanXy  Conservative state vector [rho, rhoU, ..., rhoE, rho*k, rho*epsilon].
     * @param DiffUxy  Gradient tensor of conservative variables, size dim × nVars.
     * @param muf      Laminar (molecular) dynamic viscosity.
     * @param d        Wall distance (unused in this model but kept for interface consistency).
     * @return         Turbulent dynamic viscosity mu_t (>= 0).
     */
    template <int dim, class TU, class TDiffU>
    real GetMut_RealizableKe(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, real d)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        static const auto verySmallReal_3 = std::pow(verySmallReal, 1. / 3);
        static const auto verySmallReal_4 = std::pow(verySmallReal, 1. / 4);
        real cmu = 0.09;
        real phi = 2. / 3.;
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SS = diffU + diffU.transpose() - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity();
        real rho = UMeanXy(0);
        real k = UMeanXy(I4 + 1) / rho + verySmallReal_4;
        real epsilon = UMeanXy(I4 + 2) / rho + verySmallReal_4;
        real Ret = rho * sqr(k) / (muf * epsilon) + smallReal;
        real S = std::sqrt(SS.squaredNorm() / 2) + verySmallReal_4;
        real fmu = (1 - std::exp(-0.01 * Ret)) / (1 - exp(-std::sqrt(Ret))) * std::max(1., std::sqrt(2 / Ret));
        real mut = cmu * fmu * rho * sqr(k) / epsilon;
        mut = std::min(mut, phi * rho * k / S);
#if KE_LIMIT_MUT == 1
        mut = std::min(mut, 1e5 * muf); // CFL3D
#endif
        if (std::isnan(mut) || !std::isfinite(mut))
        {
            std::cerr << k << " " << epsilon << " " << Ret << " " << S << "\n";
            std::cerr << fmu << "\n";
            std::cerr << SS << std::endl;
            DNDS_assert(false);
        }
        return mut;
    }

    /**
     * @brief Compute the RANS viscous flux for the realizable k-epsilon transport equations.
     *
     * Evaluates the diffusion (viscous) flux contributions for the k and epsilon equations
     * projected onto the face normal direction. The effective diffusivities are:
     * - k-equation:       (mu_laminar + mu_t / sigma_k),  sigma_k = 1.0
     * - epsilon-equation: (mu_laminar + mu_t / sigma_e),  sigma_e = 1.3
     *
     * @tparam dim      Spatial dimension (2 or 3).
     * @tparam TU       Eigen-compatible type for the conservative state vector.
     * @tparam TN       Eigen-compatible type for the face unit normal vector.
     * @tparam TDiffU   Eigen-compatible type for the gradient tensor (dim × nVars, primitive variables).
     * @tparam TVFlux   Eigen-compatible type for the output viscous flux vector.
     * @param UMeanXy      Conservative state vector.
     * @param DiffUxyPrim  Gradient tensor of *primitive* variables, size dim × nVars.
     * @param uNorm        Outward-pointing face unit normal vector (length dim).
     * @param mut          Turbulent dynamic viscosity (precomputed).
     * @param d            Wall distance (unused here, kept for interface consistency).
     * @param muPhy        Laminar (molecular) dynamic viscosity.
     * @param[out] vFlux   Output viscous flux vector; entries at I4+1 and I4+2 are set.
     */
    template <int dim, class TU, class TN, class TDiffU, class TVFlux>
    void GetVisFlux_RealizableKe(TU &&UMeanXy, TDiffU &&DiffUxyPrim, TN &&uNorm, real mut, real d, real muPhy, TVFlux &vFlux)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        real sigK = 1.;
        real sigE = 1.3;
        vFlux(I4 + 1) = DiffUxyPrim(Seq012, {I4 + 1}).dot(uNorm) * (muPhy + mut / sigK);
        vFlux(I4 + 2) = DiffUxyPrim(Seq012, {I4 + 2}).dot(uNorm) * (muPhy + mut / sigE);
    }

    /**
     * @brief Compute source terms for the realizable k-epsilon transport equations.
     *
     * Evaluates production, destruction, and extra source contributions for the
     * k and epsilon equations following the Shih et al. realizable k-epsilon model.
     *
     * Key terms computed:
     * - Reynolds stress tensor tau_{ij} via Boussinesq hypothesis with realizability clipping.
     * - Production of k: P_k = -tau_{ij} * dU_i/dx_j, capped by the Shih pphi limiter.
     * - Turbulent time scale T_t with low-Re correction via zeta = sqrt(Re_t / 2).
     * - Extra dissipation term E involving the gradient of the turbulent time scale.
     *
     * When @p mode == 0, the full source vector is returned:
     * - source(I4+1) = P_k - rho*epsilon
     * - source(I4+2) = (c_{e1} * P_k - c_{e2} * rho*epsilon + E) / T_t
     *
     * When @p mode == 1, the implicit diagonal contribution (positive) is returned for
     * use in implicit time stepping:
     * - source(I4+1) = 0
     * - source(I4+2) = c_{e2} / T_t
     *
     * @tparam dim      Spatial dimension (2 or 3).
     * @tparam TU       Eigen-compatible type for the conservative state vector.
     * @tparam TDiffU   Eigen-compatible type for the gradient tensor (dim × nVars, conservative variables).
     * @tparam TSource  Eigen-compatible type for the output source vector.
     * @param UMeanXy   Conservative state vector.
     * @param DiffUxy   Gradient tensor of conservative variables, size dim × nVars.
     * @param muf       Laminar (molecular) dynamic viscosity.
     * @param d         Wall distance (unused here, kept for interface consistency).
     * @param[out] source  Output source vector; entries at I4+1 and I4+2 are set.
     * @param mode      Source computation mode: 0 = full source, 1 = implicit diagonal (destruction only).
     */
    template <int dim, class TU, class TDiffU, class TSource>
    void GetSource_RealizableKe(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, real d, TSource &source, int mode)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        static const auto verySmallReal_3 = std::pow(verySmallReal, 1. / 3);
        static const auto verySmallReal_4 = std::pow(verySmallReal, 1. / 4);

        real cmu = 0.09;
        real phi = 2. / 3.;
        real ce1 = 1.44;
        real ce2 = 1.92;
        real AE = 0.3;
        real pphi = 1.065;
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SS = diffU + diffU.transpose() - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity();
        real rho = UMeanXy(0);
        real k = std::max(UMeanXy(I4 + 1) / rho, verySmallReal_4);
        real epsilon = std::max(UMeanXy(I4 + 1) / rho, verySmallReal_4);
        real Ret = rho * sqr(k) / (muf * epsilon) + verySmallReal_4;
        real S = std::sqrt(SS.squaredNorm() / 2) + verySmallReal_4;
        real fmu = (1 - std::exp(-0.01 * Ret)) / std::max(1 - exp(-std::sqrt(Ret)), verySmallReal_4) * std::max(1., std::sqrt(2 / Ret));
        real mut = cmu * fmu * rho * sqr(k) / epsilon;
        mut = std::min(mut, phi * rho * k / S);

        Eigen::Matrix<real, dim, dim> rhoMuiuj = Eigen::Matrix<real, dim, dim>::Identity() * UMeanXy(I4 + 1) * (2. / 3.) - mut * SS;
        real Pk = -(rhoMuiuj.array() * diffU.array()).sum();
        Pk = std::min(Pk, pphi * cmu * sqr(UMeanXy(I4 + 1)) / mut);
        real zeta = std::sqrt(Ret / 2);
        real Tt = k / epsilon * std::max(1., 1. / zeta) + smallReal;

        Eigen::Matrix<real, dim, 2> diffRhoKe = DiffUxy(Seq012, {I4 + 1, I4 + 2});
        Eigen::Matrix<real, dim, 2> diffKe = (diffRhoKe - 1. / UMeanXy(0) * diffRho * UMeanXy({I4 + 1, I4 + 2}).transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffTau = diffKe(Seq012, 0) / epsilon - k / sqr(epsilon) * diffKe(Seq012, 1);
        real Psi = std::max(0., diffKe(Seq012, 0).dot(diffTau));
        real E = AE * rho * std::sqrt(epsilon * Tt) * Psi * std::max(std::sqrt(k), std::pow(muf * epsilon / rho, 0.25));

        if (mode == 0)
        {
            source(I4 + 1) = Pk - UMeanXy(I4 + 2);
            source(I4 + 2) = (ce1 * Pk - ce2 * UMeanXy(I4 + 2) + E) / Tt;
        }
        else
        {
            source(I4 + 1) = 0;
            source(I4 + 2) = (ce2) / Tt;
        }
        if (!source.allFinite() || source.hasNaN())
        {
            std::cerr << source.transpose() << "\n";
            std::cerr << UMeanXy.transpose() << "\n";
            std::cerr << DiffUxy << "\n";
            std::cerr << S << "\n";
            std::cerr << mut << "\n";

            std::cout << std::endl;

            DNDS_assert(false);
        }
    }

    /**
     * @brief Attempt to find equilibrium epsilon for zero-gradient (freestream) conditions via Newton iteration.
     *
     * Iteratively solves for the epsilon value that zeroes the k-equation source term
     * when all spatial gradients are zero.  Uses a simple Newton-Raphson loop with
     * finite-difference Jacobian evaluation.
     *
     * @warning This function is noted as **not applicable** in practice — the zero-gradient
     *          equilibrium assumption does not hold for the realizable k-epsilon model.
     *          Retained for reference and experimentation only.
     *
     * @tparam dim  Spatial dimension (2 or 3).
     * @tparam TU   Eigen-compatible type for the conservative state vector.
     * @param[in,out] u      Conservative state vector; u(I4+2) (rho*epsilon) is updated in place.
     * @param         muPhy  Laminar (molecular) dynamic viscosity.
     * @return A tuple (rhs_initial, rhs_final) giving the k-equation residual before and
     *         after the Newton iteration, for convergence diagnostics.
     */
    template <int dim, class TU>
    std::tuple<real, real> SolveZeroGradEquilibrium(TU &u, real muPhy) // this is tested to be not applicable!
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        Eigen::Matrix<real, dim, 7> diffU;
        diffU.setZero();

        Eigen::Vector<real, Eigen::Dynamic> src = u;
        Eigen::Vector<real, Eigen::Dynamic> uc = u;
        src.setZero();

        real distEps = std::sqrt(smallReal);
        real distV = 1 + distEps;

        auto getDE = [&](real re) -> real
        {
            uc(I4 + 2) = re;
            GetSource_RealizableKe<dim>(uc, diffU, muPhy, src);
            return src(I4 + 1);
        };
        std::cout << "Mu " << muPhy << std::endl;

        real re = u(I4 + 2);
        real rhs0{0}, rhs{0};
        for (int i = 0; i < 1000; i++)
        {
            rhs = getDE(re);
            real dRhsDRe = (getDE(re * distV) - rhs) / (re * distEps);
            real ren = re - rhs / (dRhsDRe + verySmallReal);
            re = std::max(ren, verySmallReal);
            std::cout << rhs << std::endl;
            if (i == 0)
                rhs0 = rhs;
        }
        u(I4 + 2) = re;
        return std::make_tuple(rhs0, rhs);
    }

    /**
     * @brief Compute the analytical diagonal of the source Jacobian for the realizable k-epsilon model.
     *
     * Evaluates -dS/dU diagonal entries for implicit time stepping of the k and epsilon
     * transport equations.  The diagonal entries approximate the linearised destruction terms:
     * - source(I4+1) = -P_k / (rho*k)  (destruction rate of k)
     * - source(I4+2) = c_{e2} / T_t     (destruction rate of epsilon)
     *
     * These positive values are used to augment the implicit operator diagonal,
     * improving stability of the coupled RANS system.
     *
     * @tparam dim      Spatial dimension (2 or 3).
     * @tparam TU       Eigen-compatible type for the conservative state vector.
     * @tparam TDiffU   Eigen-compatible type for the gradient tensor (dim × nVars, conservative variables).
     * @tparam TSource  Eigen-compatible type for the output Jacobian diagonal vector.
     * @param UMeanXy   Conservative state vector.
     * @param DiffUxy   Gradient tensor of conservative variables, size dim × nVars.
     * @param muf       Laminar (molecular) dynamic viscosity.
     * @param[out] source  Output Jacobian diagonal vector; entries at I4+1 and I4+2 are set.
     */
    template <int dim, class TU, class TDiffU, class TSource>
    void GetSourceJacobianDiag_RealizableKe(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, TSource &source)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;

        real cmu = 0.09;
        real phi = 2. / 3.;
        real ce1 = 1.44;
        real ce2 = 1.92;
        real AE = 0.3;
        real pphi = 1.065;
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SS = diffU + diffU.transpose() - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity();
        real rho = UMeanXy(0);
        real k = UMeanXy(I4 + 1) / rho + verySmallReal;
        real epsilon = UMeanXy(I4 + 2) / rho + verySmallReal;
        real Ret = rho * sqr(k) / (muf * epsilon) + verySmallReal;
        real S = std::sqrt(SS.squaredNorm() / 2) + verySmallReal;
        real fmu = (1 - std::exp(-0.01 * Ret)) / (1 - exp(-std::sqrt(Ret))) * std::max(1., std::sqrt(2 / Ret));
        real mut = cmu * fmu * rho * sqr(k) / epsilon;
        mut = std::min(mut, phi * rho * k / S);

        Eigen::Matrix<real, dim, dim> rhoMuiuj = Eigen::Matrix<real, dim, dim>::Identity() * UMeanXy(I4 + 1) * (2. / 3.) - mut * SS;
        real Pk = -(rhoMuiuj.array() * diffU.array()).sum();
        Pk = std::min(Pk, pphi * cmu * sqr(UMeanXy(I4 + 1)) / mut);
        real zeta = std::sqrt(Ret / 2);
        real Tt = k / epsilon * std::max(1., 1. / zeta) + smallReal;

        Eigen::Matrix<real, dim, 2> diffRhoKe = DiffUxy(Seq012, {I4 + 1, I4 + 2});
        Eigen::Matrix<real, dim, 2> diffKe = (diffRhoKe - 1. / UMeanXy(0) * diffRho * UMeanXy({I4 + 1, I4 + 2}).transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffTau = diffKe(Seq012, 0) / epsilon - k / sqr(epsilon) * diffKe(Seq012, 1);
        real Psi = std::max(0., diffKe(Seq012, 0).dot(diffTau));
        real E = AE * rho * std::sqrt(epsilon * Tt) * Psi * std::max(std::sqrt(k), std::pow(muf * epsilon / rho, 0.25));

        real dPk = -((Eigen::Matrix<real, dim, dim>::Identity() * (2. / 3.)).array() * diffU.array()).sum();

        source(I4 + 1) = -Pk / (UMeanXy(I4 + 1) + verySmallReal);
        source(I4 + 2) = (ce2) / Tt;
    }

    /**
     * @brief Compute the numerical (finite-difference) diagonal of the source Jacobian for the realizable k-epsilon model.
     *
     * Evaluates -dS_i/dU_i by forward finite-difference perturbation of rho*k and rho*epsilon
     * independently.  The perturbation step is proportional to sqrt(smallReal) times the variable
     * magnitude.  Results are clamped to non-negative values and amplified by a factor of 10
     * for enhanced implicit stability.
     *
     * This is a fallback when the analytical Jacobian diagonal
     * (GetSourceJacobianDiag_RealizableKe) is suspected of inaccuracy.
     *
     * @tparam dim      Spatial dimension (2 or 3).
     * @tparam TU       Eigen-compatible type for the conservative state vector.
     * @tparam TDiffU   Eigen-compatible type for the gradient tensor (dim × nVars, conservative variables).
     * @tparam TSource  Eigen-compatible type for the output Jacobian diagonal vector.
     * @param UMeanXy   Conservative state vector.
     * @param DiffUxy   Gradient tensor of conservative variables, size dim × nVars.
     * @param muf       Laminar (molecular) dynamic viscosity.
     * @param[out] source  Output numerical Jacobian diagonal vector; entries at I4+1 and I4+2 are set.
     */
    template <int dim, class TU, class TDiffU, class TSource>
    void GetSourceJacobianDiag_RealizableKe_ND(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, TSource &source)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        Eigen::VectorXd u0 = UMeanXy;
        Eigen::VectorXd u1 = UMeanXy;
        Eigen::VectorXd sb = source;
        Eigen::VectorXd s0 = source;
        Eigen::VectorXd s1 = source;
        real epsRk = (u0(I4 + 1) + smallReal) * std::sqrt(smallReal);
        real epsRe = (u0(I4 + 2) + smallReal) * std::sqrt(smallReal);
        u0(I4 + 1) += epsRk;
        u1(I4 + 2) += epsRe;
        GetSource_RealizableKe<dim>(UMeanXy, DiffUxy, muf, sb);
        GetSource_RealizableKe<dim>(u0, DiffUxy, muf, s0);
        GetSource_RealizableKe<dim>(u1, DiffUxy, muf, s1);
        source(I4 + 1) = -(s0(I4 + 1) - sb(I4 + 1)) / epsRk;
        source(I4 + 2) = -(s1(I4 + 2) - sb(I4 + 2)) / epsRe;
        source(I4 + 1) = std::max(source(I4 + 1), 0.) * 10;
        source(I4 + 2) = std::max(source(I4 + 2), 0.) * 10;
    }

    /**
     * @brief Compute turbulent viscosity mu_t from Menter's k-omega SST model.
     *
     * Evaluates the SST eddy viscosity with:
     * - Vorticity magnitude Omega = ||(grad U)^T - grad U|| / sqrt(2).
     * - Blending function F2 based on wall distance and turbulent Reynolds number.
     * - Bradshaw's structural constraint: mu_t = a1 * k / max(Omega * F2, a1 * omega),
     *   where a1 = 0.31.
     * - Optional upper bound of 1e5 * mu_laminar when @c KW_SST_LIMIT_MUT is enabled.
     *
     * @tparam dim     Spatial dimension (2 or 3).
     * @tparam TU      Eigen-compatible type for the conservative state vector.
     * @tparam TDiffU  Eigen-compatible type for the gradient tensor (dim × nVars, conservative variables).
     * @param UMeanXy  Conservative state vector [rho, rhoU, ..., rhoE, rho*k, rho*omega].
     * @param DiffUxy  Gradient tensor of conservative variables, size dim × nVars.
     * @param muf      Laminar (molecular) dynamic viscosity.
     * @param d        Wall distance.
     * @return         Turbulent dynamic viscosity mu_t (>= 0).
     */
    template <int dim, class TU, class TDiffU>
    real GetMut_SST(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, real d)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        static const auto verySmallReal_3 = std::pow(verySmallReal, 1. / 3);
        static const auto verySmallReal_4 = std::pow(verySmallReal, 1. / 4);

        real a1 = 0.31;
        real betaStar = 0.09;
        real sigOmega2 = 0.856;
        real cmu = 0.09;
        real phi = 2. / 3.;
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SR2 = diffU + diffU.transpose();                                                  // 2 times strain rate
        Eigen::Matrix<real, dim, dim> SS = SR2 - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity(); // 2 times shear strain rate
        Eigen::Matrix<real, dim, dim> OmegaM = (diffU.transpose() - diffU) * 0.5;
        real OmegaMag = OmegaM.norm() * std::sqrt(2);
        real rho = UMeanXy(0);
        real k = std::max(UMeanXy(I4 + 1) / rho, verySmallReal_4);
        real omegaaa = std::max(UMeanXy(I4 + 2) / rho, verySmallReal_4);
        real S = std::sqrt(SS.squaredNorm() / 2) + verySmallReal_4;
        real nuPhy = muf / rho;
        real F2 = std::tanh(sqr(std::max(2 * std::sqrt(k) / (betaStar * omegaaa * d), 500 * nuPhy / (sqr(d) * omegaaa))));
        // F2 = 0;
        real mut = a1 * k / std::max(OmegaMag * F2, a1 * omegaaa) * rho;
#if KW_SST_LIMIT_MUT == 1
        mut = std::min(mut, 1e5 * muf); // CFL3D
#endif

        if (std::isnan(mut) || !std::isfinite(mut))
        {
            std::cerr << k << " " << omegaaa << " " << mut << " " << S << "\n";
            std::cerr << SS << std::endl;
            DNDS_assert(false);
        }
        return mut;
    }

    /**
     * @brief Compute the RANS viscous flux for the k-omega SST transport equations.
     *
     * Evaluates the diffusion (viscous) flux contributions for the k and omega equations
     * projected onto the face normal direction.  The effective diffusion coefficients are
     * blended between the inner-layer (set 1) and outer-layer (set 2) values using the
     * SST blending function F1:
     * - sigma_k = sigma_k1 * F1 + sigma_k2 * (1 - F1)
     * - sigma_omega = sigma_omega1 * F1 + sigma_omega2 * (1 - F1)
     *
     * Effective diffusivity for each equation: mu_laminar + mu_t * sigma_{k,omega}.
     *
     * @tparam dim      Spatial dimension (2 or 3).
     * @tparam TU       Eigen-compatible type for the conservative state vector.
     * @tparam TN       Eigen-compatible type for the face unit normal vector.
     * @tparam TDiffU   Eigen-compatible type for the gradient tensor (dim × nVars, primitive variables).
     * @tparam TVFlux   Eigen-compatible type for the output viscous flux vector.
     * @param UMeanXy      Conservative state vector.
     * @param DiffUxyPrim  Gradient tensor of *primitive* variables, size dim × nVars.
     * @param uNorm        Outward-pointing face unit normal vector (length dim).
     * @param mutIn        Turbulent dynamic viscosity (precomputed, used for diffusion coefficients).
     * @param d            Wall distance.
     * @param muf          Laminar (molecular) dynamic viscosity.
     * @param[out] vFlux   Output viscous flux vector; entries at I4+1 and I4+2 are set.
     */
    template <int dim, class TU, class TN, class TDiffU, class TVFlux>
    void GetVisFlux_SST(TU &&UMeanXy, TDiffU &&DiffUxyPrim, TN &&uNorm, real mutIn, real d, real muf, TVFlux &vFlux)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        static const auto verySmallReal_3 = std::pow(verySmallReal, 1. / 3);
        static const auto verySmallReal_4 = std::pow(verySmallReal, 1. / 4);

        real a1 = 0.31;
        real betaStar = 0.09;
        real sigK1 = 0.85;
        real sigK2 = 1;
        real sigO1 = 0.5;
        real sigO2 = 0.856;
        real beta1 = 0.075;
        real beta2 = 0.0828;
        real kap = 0.41;
        real gamma1 = beta1 / betaStar - sigO1 * sqr(kap) / std::sqrt(betaStar);
        real gamma2 = beta2 / betaStar - sigO2 * sqr(kap) / std::sqrt(betaStar);
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxyPrim(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffU = DiffUxyPrim(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> SR2 = diffU + diffU.transpose();                                                  // 2 times strain rate
        Eigen::Matrix<real, dim, dim> SS = SR2 - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity(); // 2 times shear strain rate
        Eigen::Matrix<real, dim, 2> diffKO = DiffUxyPrim(Seq012, {I4 + 1, I4 + 2});

        Eigen::Matrix<real, dim, dim> OmegaM = (diffU.transpose() - diffU) * 0.5;
        real OmegaMag = OmegaM.norm() * std::sqrt(2);
        real rho = UMeanXy(0);
        real k = std::max(UMeanXy(I4 + 1) / rho, verySmallReal_4);
        real omegaaa = std::max(UMeanXy(I4 + 2) / rho, verySmallReal_4);
        real S = std::sqrt(SS.squaredNorm() / 2) + verySmallReal_4;
        real nuPhy = muf / rho;
        real CDKW = std::max(2 * rho * sigO2 / omegaaa * diffKO(EigenAll, 0).dot(diffKO(EigenAll, 1)), 1e-10);
        real F1 = std::tanh(std::pow(
            std::min(std::max(std::sqrt(k) / (betaStar * omegaaa * d), 500 * nuPhy / (sqr(d) * omegaaa)),
                     4 * rho * sigO2 * k / (CDKW * sqr(d))),
            4));
        real F2 = std::tanh(sqr(std::max(2 * std::sqrt(k) / (betaStar * omegaaa * d), 500 * nuPhy / (sqr(d) * omegaaa))));
        // F2 = 0;
        real mut = a1 * k / std::max(OmegaMag * F2, a1 * omegaaa) * rho;
#if KW_SST_LIMIT_MUT == 1
        mut = std::min(mut, 1e5 * muf); // CFL3D
#endif

        real sigK = sigK1 * F1 + sigK2 * (1 - F1);
        real sigO = sigO1 * F1 + sigO2 * (1 - F1);

        vFlux(I4 + 1) = diffKO(Seq012, 0).dot(uNorm) * (muf + mutIn * sigK);
        vFlux(I4 + 2) = diffKO(Seq012, 1).dot(uNorm) * (muf + mutIn * sigO);

        if (!vFlux.allFinite() || vFlux.hasNaN())
        {
            std::cerr << vFlux << "\n";
            std::cerr << sigK << " " << sigO << "\n";
            std::cerr << F1 << " " << F2 << "\n";
            std::cerr << CDKW << "\n";
            std::cerr << k << " " << omegaaa << "\n";
            std::cerr << muf << " " << mut << "\n";
            std::cerr << std::endl;
        }
    }

    /**
     * @brief Compute source terms for the k-omega SST transport equations.
     *
     * Evaluates production, destruction, and cross-diffusion source contributions for
     * the k and omega equations following Menter's SST model.  Supports DES/DDES
     * capability via the RANS-to-LES length scale ratio.
     *
     * Key terms:
     * - Production P_k from the Boussinesq Reynolds stress, optionally capped at
     *   20 * beta* * rho * k * omega (when @c KW_SST_PROD_LIMITS is enabled).
     * - Omega production P_omega with version-dependent formulation controlled by
     *   @c KW_SST_PROD_OMEGA_VERSION (0 = classical, 1 = strain-rate, 2 = vorticity).
     * - Destruction terms: beta* * rho * k * omega for k, beta * rho * omega^2 for omega.
     * - Cross-diffusion term: 2 * (1 - F1) * rho * sigma_omega2 / omega * grad(k) . grad(omega).
     * - DES shielding: F_DES = max(1, l_RANS / l_LES * (1 - F2)), reducing the destruction
     *   of k outside the RANS region.
     *
     * When @p mode == 0, the full source vector is returned.
     * When @p mode == 1, the implicit diagonal contribution (positive) is returned:
     * - source(I4+1) = beta* * omega * F_DES
     * - source(I4+2) = 2 * beta * omega
     *
     * @tparam dim      Spatial dimension (2 or 3).
     * @tparam TU       Eigen-compatible type for the conservative state vector.
     * @tparam TDiffU   Eigen-compatible type for the gradient tensor (dim × nVars, conservative variables).
     * @tparam TSource  Eigen-compatible type for the output source vector.
     * @param UMeanXy   Conservative state vector.
     * @param DiffUxy   Gradient tensor of conservative variables, size dim × nVars.
     * @param muf       Laminar (molecular) dynamic viscosity.
     * @param d         Wall distance.
     * @param lLES      LES length scale for DES/DDES shielding (set to a large value to disable DES).
     * @param[out] source  Output source vector; entries at I4+1 and I4+2 are set.
     * @param mode      Source computation mode: 0 = full source, 1 = implicit diagonal (destruction only).
     */
    template <int dim, class TU, class TDiffU, class TSource>
    void GetSource_SST(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, real d, real lLES, TSource &source, int mode)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        static const auto verySmallReal_3 = std::pow(verySmallReal, 1. / 3);
        static const auto verySmallReal_4 = std::pow(verySmallReal, 1. / 4);

        real a1 = 0.31;
        real betaStar = 0.09;
        real sigK1 = 0.85;
        real sigK2 = 1;
        real sigO1 = 0.5;
        real sigO2 = 0.856;
        real beta1 = 0.075;
        real beta2 = 0.0828;
        real kap = 0.41;
        real gamma1 = beta1 / betaStar - sigO1 * sqr(kap) / std::sqrt(betaStar);
        real gamma2 = beta2 / betaStar - sigO2 * sqr(kap) / std::sqrt(betaStar);
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SR2 = diffU + diffU.transpose();                                                  // 2 times strain rate
        Eigen::Matrix<real, dim, dim> SS = SR2 - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity(); // 2 times shear strain rate
        Eigen::Matrix<real, dim, 2> diffRhoKO = DiffUxy(Seq012, {I4 + 1, I4 + 2});
        Eigen::Matrix<real, dim, 2> diffKO = (diffRhoKO - 1. / UMeanXy(0) * diffRho * UMeanXy({I4 + 1, I4 + 2}).transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> OmegaM = (diffU.transpose() - diffU) * 0.5;
        real OmegaMag = OmegaM.norm() * std::sqrt(2);
        real rho = UMeanXy(0);
        real k = std::max(UMeanXy(I4 + 1) / rho, verySmallReal_4);
        real omegaaa = std::max(UMeanXy(I4 + 2) / rho, verySmallReal_4);
        real S = std::sqrt(SS.squaredNorm() / 2) + verySmallReal_4;
        real nuPhy = muf / rho;
        real CDKW = std::max(2 * rho * sigO2 / omegaaa * diffKO(EigenAll, 0).dot(diffKO(EigenAll, 1)), 1e-10);
        real F1 = std::tanh(std::pow(
            std::min(std::max(std::sqrt(k) / (betaStar * omegaaa * d), 500 * nuPhy / (sqr(d) * omegaaa)),
                     4 * rho * sigO2 * k / (CDKW * sqr(d))),
            4));
        real F2 = std::tanh(sqr(std::max(2 * std::sqrt(k) / (betaStar * omegaaa * d), 500 * nuPhy / (sqr(d) * omegaaa))));
        real lRANS = std::sqrt(k) / (betaStar * omegaaa);
        real FDES = std::max(1.0, lRANS / (lLES + verySmallReal) * (1 - F2));
        // F2 = 0;
        real mut = a1 * k / std::max(OmegaMag * F2, a1 * omegaaa) * rho; // use S/OmegaMag for SST: S: CFD++, OmegaMag: Turbulence Modeling Validation, Testing, and Developmen
#if KW_SST_LIMIT_MUT == 1
        mut = std::min(mut, 1e5 * muf); // CFL3D
#endif
        real nutHat = std::max(mut / rho, 1e-8);

        Eigen::Matrix<real, dim, dim> rhoMuiuj = Eigen::Matrix<real, dim, dim>::Identity() * UMeanXy(I4 + 1) * (2. / 3.) - mut * SS;
        real Pk = -(rhoMuiuj.array() * diffU.array()).sum();
        real PkTilde = Pk;
#if KW_SST_PROD_LIMITS == 1
        PkTilde = std::max(PkTilde, verySmallReal);
        PkTilde = std::min(Pk, 20 * betaStar * rho * k * omegaaa); // CFD++'s limiting: 10 times
#endif

        real gammaC = gamma1 * F1 + gamma1 * (1 - F1);
        real sigK = sigK1 * F1 + sigK2 * (1 - F1);
        real sigO = sigO1 * F1 + sigO2 * (1 - F1);
        real beta = beta1 * F1 + beta2 * (1 - F1);
        real POmega = gammaC / nutHat * Pk;
#if KW_SST_PROD_OMEGA_VERSION == 1
        POmega =
            0.5 * gammaC * rho * ((SR2 - SR2.trace() / 3. * Eigen::Matrix<real, dim, dim>::Identity()).array() * SR2.array()).sum();
#elif KW_SST_PROD_OMEGA_VERSION == 2
        POmega =
            gammaC * rho * sqr(OmegaMag);
#endif
        if (mode == 0)
        {
            source(I4 + 1) = PkTilde - betaStar * rho * k * omegaaa * FDES;
            source(I4 + 2) = POmega - beta * rho * sqr(omegaaa) +
                             2 * (1 - F1) * rho * sigO2 / omegaaa * diffKO(EigenAll, 0).dot(diffKO(EigenAll, 1));
            // source(I4 + 2) = POmega - beta * rho * sqr(omegaaa) +
            //                  2 * (1 - F1) * rho * sigO2 / omegaaa * diffKO(EigenAll, 0).dot(diffKO(EigenAll, 1));
        }
        else
        {
            source(I4 + 1) = betaStar * omegaaa * FDES;
            source(I4 + 2) = 2 * beta * omegaaa;
        }
    }

    /**
     * @brief Compute turbulent viscosity mu_t from the Wilcox k-omega model.
     *
     * Evaluates the eddy viscosity mu_t = rho * k / omega_tilde, where omega_tilde
     * depends on the selected model version (@c KW_WILCOX_VER):
     * - Version 0 (original 1988): omega_tilde = omega (no stress limiter).
     * - Version 1 (2006): omega_tilde = max(omega, C_lim * sqrt(S_{ij}S_{ij} / beta*)),
     *   applying the stress limiter with C_lim = 7/8.
     * - Version 2 (simplified): omega_tilde = omega (no stress limiter).
     *
     * Optional upper bound of 1e5 * mu_laminar when @c KW_WILCOX_LIMIT_MUT is enabled.
     *
     * @tparam dim     Spatial dimension (2 or 3).
     * @tparam TU      Eigen-compatible type for the conservative state vector.
     * @tparam TDiffU  Eigen-compatible type for the gradient tensor (dim × nVars, conservative variables).
     * @param UMeanXy  Conservative state vector [rho, rhoU, ..., rhoE, rho*k, rho*omega].
     * @param DiffUxy  Gradient tensor of conservative variables, size dim × nVars.
     * @param muf      Laminar (molecular) dynamic viscosity.
     * @param d        Wall distance (unused in this model but kept for interface consistency).
     * @return         Turbulent dynamic viscosity mu_t (>= 0).
     */
    template <int dim, class TU, class TDiffU>
    real GetMut_KOWilcox(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, real d)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        static const auto verySmallReal_3 = std::pow(verySmallReal, 1. / 3);
        static const auto verySmallReal_4 = std::pow(verySmallReal, 1. / 4);

        real CLim = 7. / 8.;
        real betaS = 0.09;
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SR2 = diffU + diffU.transpose();
        real rho = UMeanXy(0);
        real k = std::max(UMeanXy(I4 + 1) / rho, sqr(verySmallReal_4));
        real omegaaa = std::max(UMeanXy(I4 + 2) / rho, verySmallReal_4);
#if KW_WILCOX_VER == 0
        real omegaaaTut = omegaaa;
#else
        real omegaaaTut = std::max(omegaaa, CLim * std::sqrt(0.5 * SR2.squaredNorm() / betaS));
#endif
        real mut = k / omegaaaTut * rho;
#if KW_WILCOX_LIMIT_MUT == 1
        mut = std::min(mut, 1e5 * muf); // CFL3D
#endif

        if (std::isnan(mut) || !std::isfinite(mut))
        {
            std::cerr << k << " " << omegaaa << " " << mut << "\n";
            DNDS_assert(false);
        }
        return mut;
    }

    /**
     * @brief Compute the RANS viscous flux for the Wilcox k-omega transport equations.
     *
     * Evaluates the diffusion (viscous) flux contributions for the k and omega equations
     * projected onto the face normal direction.  The Wilcox model uses constant diffusion
     * coefficients:
     * - sigma_k = 0.5
     * - sigma_omega = 0.5
     *
     * Effective diffusivity: mu_laminar + mu_t * sigma_{k,omega}.
     *
     * @tparam dim      Spatial dimension (2 or 3).
     * @tparam TU       Eigen-compatible type for the conservative state vector.
     * @tparam TN       Eigen-compatible type for the face unit normal vector.
     * @tparam TDiffU   Eigen-compatible type for the gradient tensor (dim × nVars, primitive variables).
     * @tparam TVFlux   Eigen-compatible type for the output viscous flux vector.
     * @param UMeanXy      Conservative state vector.
     * @param DiffUxyPrim  Gradient tensor of *primitive* variables, size dim × nVars.
     * @param uNorm        Outward-pointing face unit normal vector (length dim).
     * @param mutIn        Turbulent dynamic viscosity (precomputed).
     * @param d            Wall distance (unused here, kept for interface consistency).
     * @param muf          Laminar (molecular) dynamic viscosity.
     * @param[out] vFlux   Output viscous flux vector; entries at I4+1 and I4+2 are set.
     */
    template <int dim, class TU, class TN, class TDiffU, class TVFlux>
    void GetVisFlux_KOWilcox(TU &&UMeanXy, TDiffU &&DiffUxyPrim, TN &&uNorm, real mutIn, real d, real muf, TVFlux &vFlux)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        static const auto verySmallReal_3 = std::pow(verySmallReal, 1. / 3);
        static const auto verySmallReal_4 = std::pow(verySmallReal, 1. / 4);

        real alpha = 13. / 25.;
        real betaS = 0.09;
        real sigK = 0.5;
        real sigO = 0.5;
        real Prt = 0.9;
        real CLim = 7. / 8.;
        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxyPrim(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffU = DiffUxyPrim(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> SR2 = diffU + diffU.transpose();                                                  // 2 times strain rate
        Eigen::Matrix<real, dim, dim> SS = SR2 - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity(); // 2 times shear strain rate
        Eigen::Matrix<real, dim, 2> diffKO = DiffUxyPrim(Seq012, {I4 + 1, I4 + 2});
        real rho = UMeanXy(0);
        real k = std::max(UMeanXy(I4 + 1) / rho, sqr(verySmallReal_4));
        real omegaaa = std::max(UMeanXy(I4 + 2) / rho, verySmallReal_4);
#if KW_WILCOX_VER == 0
        real omegaaaTut = omegaaa;
#else
        real omegaaaTut = std::max(omegaaa, CLim * std::sqrt(0.5 * SR2.squaredNorm() / betaS));
#endif
        real mut = k / omegaaaTut * rho;
#if KW_WILCOX_LIMIT_MUT == 1
        mut = std::min(mut, 1e5 * muf); // CFL3D
#endif

        vFlux(I4 + 1) = diffKO(Seq012, 0).dot(uNorm) * (muf + mut * sigK);
        vFlux(I4 + 2) = diffKO(Seq012, 1).dot(uNorm) * (muf + mut * sigO);
    }

    /**
     * @brief Compute source terms for the Wilcox k-omega transport equations.
     *
     * Evaluates production, destruction, and cross-diffusion source contributions for
     * the k and omega equations following the Wilcox k-omega model.  The version-dependent
     * behavior is controlled by @c KW_WILCOX_VER:
     *
     * - **Version 0 (1988):** alpha = 5/9, beta = 3/40, no stress limiter, no cross-diffusion.
     * - **Version 1 (2006):** alpha = 13/25, beta = f_beta * beta_0, with the f_beta correction
     *   based on Chi_omega (mean rotation / strain invariant), stress limiter C_lim = 7/8,
     *   and cross-diffusion sigma_d term (sigma_d = 0.125 when grad(k) . grad(omega) > 0).
     * - **Version 2 (simplified):** alpha = 13/25, beta = 0.075, vorticity-based production,
     *   no stress limiter, no cross-diffusion.
     *
     * Production of k is optionally capped at 20 * beta* * rho * k * omega (CFL3D-style)
     * when @c KW_WILCOX_PROD_LIMITS is enabled.
     *
     * When @p mode == 0, the full source vector is returned:
     * - source(I4+1) = P_k - beta* * rho * k * omega
     * - source(I4+2) = P_omega - beta * rho * omega^2 + sigma_d / omega * grad(k) . grad(omega)
     *
     * When @p mode == 1, the implicit diagonal contribution (positive) is returned:
     * - source(I4+1) = beta* * omega
     * - source(I4+2) = 2 * beta * omega
     *
     * @tparam dim      Spatial dimension (2 or 3).
     * @tparam TU       Eigen-compatible type for the conservative state vector.
     * @tparam TDiffU   Eigen-compatible type for the gradient tensor (dim × nVars, conservative variables).
     * @tparam TSource  Eigen-compatible type for the output source vector.
     * @param UMeanXy   Conservative state vector.
     * @param DiffUxy   Gradient tensor of conservative variables, size dim × nVars.
     * @param muf       Laminar (molecular) dynamic viscosity.
     * @param d         Wall distance (unused here, kept for interface consistency).
     * @param[out] source  Output source vector; entries at I4+1 and I4+2 are set.
     * @param mode      Source computation mode: 0 = full source, 1 = implicit diagonal (destruction only).
     */
    template <int dim, class TU, class TDiffU, class TSource>
    void GetSource_KOWilcox(TU &&UMeanXy, TDiffU &&DiffUxy, real muf, real d, TSource &source, int mode)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;
        static const auto verySmallReal_3 = std::pow(verySmallReal, 1. / 3);
        static const auto verySmallReal_4 = std::pow(verySmallReal, 1. / 4);
#if KW_WILCOX_VER == 0
        real alpha = 5. / 9.;
#else
        real alpha = 13. / 25.;
#endif
        real betaS = 0.09;
        real sigK = 0.5;
        real sigO = 0.5;
        real Prt = 0.9;
        real CLim = 7. / 8.;
        real betaO = 0.0708;
        real kappa = 0.41;

        Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);
        Eigen::Matrix<real, dim, dim> SR2 = diffU + diffU.transpose();                                                  // 2 times strain rate
        Eigen::Matrix<real, dim, dim> OmegaM2 = diffU.transpose() - diffU;                                              // 2 times rotation
        Eigen::Matrix<real, dim, dim> SS = SR2 - (2. / 3.) * diffU.trace() * Eigen::Matrix<real, dim, dim>::Identity(); // 2 times shear strain rate
        Eigen::Matrix<real, dim, 2> diffRhoKO = DiffUxy(Seq012, {I4 + 1, I4 + 2});
        Eigen::Matrix<real, dim, 2> diffKO = (diffRhoKO - 1. / UMeanXy(0) * diffRho * UMeanXy({I4 + 1, I4 + 2}).transpose()) / UMeanXy(0);
        real rho = UMeanXy(0);
        real k = std::max(UMeanXy(I4 + 1) / rho, sqr(verySmallReal_4)); // make nu -> 0 when k,O->0
        real omegaaa = std::max(UMeanXy(I4 + 2) / rho, verySmallReal_4);
#if KW_WILCOX_VER == 0 || KW_WILCOX_VER == 2
        real omegaaaTut = omegaaa;
#else
        real omegaaaTut = std::max(omegaaa, CLim * std::sqrt(0.5 * SR2.squaredNorm() / betaS));
#endif
        real mut = k / omegaaaTut * rho;
#if KW_WILCOX_LIMIT_MUT == 1
        mut = std::min(mut, 1e5 * muf); // CFL3D
#endif

        real ChiOmega = std::abs(((OmegaM2 * OmegaM2).array() * SR2.array()).sum() * 0.125 / cube(betaS * omegaaa));
        real fBeta = (1 + 85 * ChiOmega) / (1 + 100 * ChiOmega);
#if KW_WILCOX_VER == 0
        real beta = 3. / 40.;
#elif KW_WILCOX_VER == 1
        real beta = fBeta * betaO;
#else
        real beta = 0.075;
#endif
        real crossDiff = diffKO(EigenAll, 0).dot(diffKO(EigenAll, 1));
#if KW_WILCOX_VER == 0 || KW_WILCOX_VER == 2
        real SigD = 0;
#else
        real SigD = crossDiff > 0 ? 0.125 : 0;
#endif

        Eigen::Matrix<real, dim, dim> rhoMuiuj = Eigen::Matrix<real, dim, dim>::Identity() * UMeanXy(I4 + 1) * (2. / 3.) - mut * SS;
        real Pk = -(rhoMuiuj.array() * diffU.array()).sum();
        real POmega = alpha * omegaaa / k * Pk;

#if KW_WILCOX_VER == 2
        Pk = OmegaM2.squaredNorm() * 0.5 * mut;
        real gam = beta / betaS - sqr(kappa) / (sigO * std::sqrt(betaS));
        POmega = gam * rho * OmegaM2.squaredNorm() * 0.5; // CFL3D ???
#endif

#if KW_WILCOX_PROD_LIMITS == 1
        // Pk = std::min(Pk, OmegaM2.squaredNorm() * 0.5 * mut); // compare with CFL3D approx

        Pk = std::max(Pk, verySmallReal);
        Pk = std::min(Pk, betaS * rho * k * omegaaa * 20); // CFL3D
#endif

        // POmega = std::min(POmega, beta * rho * sqr(omegaaa) * 20); // CFL3D

        if (mode == 0)
        {
            source(I4 + 1) = Pk - betaS * rho * k * omegaaa;
            source(I4 + 2) = POmega - beta * rho * sqr(omegaaa) + SigD / omegaaa * crossDiff;
        }
        else
        {
            source(I4 + 1) = betaS * omegaaa;
            source(I4 + 2) = 2 * beta * omegaaa;
        }
    }

    /**
     * @brief Compute source terms for the Spalart-Allmaras one-equation turbulence model.
     *
     * Evaluates the full SA source term including production, destruction, diffusion,
     * and optional DES/DDES/IDDES/WMLES length-scale modification.  The SA model solves
     * a single transport equation for the modified kinematic viscosity nuTilde (stored
     * as rho * nuTilde at index I4+1 in the conservative state vector).
     *
     * Key terms computed:
     * - **Vorticity magnitude** S = ||Omega|| * sqrt(2), with optional rotation correction
     *   (SRotCor = c_rot * min(0, S - SS)) when @p rotCor is enabled.
     * - **Modified vorticity** Sh = S + Sbar, where Sbar = nuTilde * f_{v2} / (kappa^2 * d^2).
     * - **Production** P = c_{b1} * (1 - f_{t2}) * Sh * nuTilde.
     * - **Destruction** D = (c_{w1} * f_w - c_{b1}/kappa^2 * f_{t2}) * (nuTilde / d)^2.
     * - **Diffusion** term: c_{b2}/sigma * |grad(nuTilde)|^2 (conservative form contribution).
     * - **f_{t2} laminar suppression** term controlled by @c SA_USE_FT2_TERM.
     *
     * DES/DDES/IDDES length-scale modification:
     * - **DESMode 0 (RANS):** l_DES = d (pure RANS, wall distance).
     * - **DESMode 1 (IDDES):** Blends l_RANS and l_LES using the IDDES shielding functions
     *   f_d, f_B, f_e, with psi correction for grid-induced separation avoidance.
     * - **DESMode 2 (WMLES):** Uses the wall-modeled LES length scale
     *   l_WMLES = f_B * (1 + f_e) * l_RANS + (1 - f_B) * l_LES * psi.
     *
     * When @p mode == 0, the full source is returned at source(I4+1).
     * When @p mode == 1, the implicit diagonal contribution -dS/dU (positive, suitable for
     * augmenting the implicit operator) is returned at source(I4+1).
     *
     * @tparam dim      Spatial dimension (2 or 3).
     * @tparam TU       Eigen-compatible type for the conservative state vector.
     * @tparam TDiffU   Eigen-compatible type for the gradient tensor (dim × nVars, conservative variables).
     * @tparam TSource  Eigen-compatible type for the output source vector.
     * @param UMeanXy   Conservative state vector [rho, rhoU, ..., rhoE, rho*nuTilde].
     * @param DiffUxy   Gradient tensor of conservative variables, size dim × nVars.
     * @param muRef     Reference viscosity scale used to non-dimensionalise nuTilde
     *                  (nuTilde_physical = UMeanXy(I4+1) * muRef / rho).
     * @param mufPhy    Physical (dimensional) laminar dynamic viscosity.
     * @param gamma     Ratio of specific heats (used for pressure gradient in optional helicity correction).
     * @param d         Wall distance.
     * @param lLES      LES length scale for DES shielding (set to a large value to disable DES).
     * @param hMax      Maximum cell size (used in IDDES alpha parameter: alpha = 0.25 - d/hMax).
     * @param DESMode   DES operating mode: 0 = pure RANS, 1 = IDDES, 2 = WMLES.
     * @param[out] source  Output source vector; entry at I4+1 is set.
     * @param rotCor    Rotation correction flag: 0 = disabled, nonzero = enabled (c_rot = 2.0).
     * @param mode      Source computation mode: 0 = full source, 1 = implicit diagonal (positive).
     */
    template <int dim, class TU, class TDiffU, class TSource>
    void GetSource_SA(TU &&UMeanXy, TDiffU &&DiffUxy, real muRef, real mufPhy, real gamma,
                      real d,
                      real lLES,
                      real hMax,
                      int DESMode,
                      TSource &source, int rotCor, int mode,
                      int SAVersion = 0)
    {
        static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);
        static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
        static const auto I4 = dim + 1;

        static const real cb1 = 0.1355;
        static const real cb2 = 0.622;
        static const real sigma = SA::sigma;
        static const real cnu1 = SA::cnu1;
        static const real cnu2 = 0.7;
        static const real cnu3 = 0.9;
        static const real cw2 = 0.3;
        static const real cw3 = 2;
        static const real kappa = 0.41;
        static const real rlim = 10;
        static const real cw1 = cb1 / sqr(kappa) + (1 + cb2) / sigma;

#if SA_USE_FT2_TERM
        static const real ct3 = 1.2;
        static const real ct4 = 0.5;
#else
        static const real ct3 = 0.0;
        static const real ct4 = 0.0;
#endif

        const real nuh = UMeanXy(I4 + 1) * muRef / UMeanXy(0);

        const real Chi = (UMeanXy(I4 + 1) * muRef / mufPhy);
        const real Chi3 = cube(Chi);
        const real fnu1 = Chi3 / (Chi3 + cube(cnu1));
        const real fnu2 = 1 - Chi / (1 + Chi * fnu1);

        const Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
        const Eigen::Matrix<real, dim, 1> diffRhoNu = DiffUxy(Seq012, {I4 + 1}) * muRef;
        const Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
        const Eigen::Matrix<real, dim, 1> diffNu = (diffRhoNu - nuh * diffRho) / UMeanXy(0);
        const Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
        const Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);

        const Eigen::Matrix<real, dim, dim> Omega = 0.5 * (diffU.transpose() - diffU);
#ifndef USE_ABS_VELO_IN_ROTATION
        if (settings.frameConstRotation.enabled)
            Omega += Geom::CrossVecToMat(settings.frameConstRotation.vOmega())(Seq012, Seq012); // to static frame rotation
#endif
        const real S = Omega.norm() * std::sqrt(2.0);                               // is omega's magnitude
        const real SS = SAVersion == 0
                            ? (diffU + diffU.transpose()).norm() * (1. / std::sqrt(2.0))  // corrected strain rate magnitude
                            : (diffU + diffU.transpose()).norm();                          // legacy (Frobenius of 2*S_ij)
        real SRotCor = 0.0;
        if (rotCor)
        {
            const real cRot = SAVersion == 0 ? 2.0 : 1.0;
            SRotCor += SAVersion == 0
                           ? cRot * std::min(0.0, S - SS)
                           : cRot * std::min(0.0, SS - S);
        }
        real diffUNorm = diffU.norm();

        const real ft2 = ct3 * std::exp(-ct4 * sqr(Chi));
        // {
        //     Eigen::Matrix<real, dim, dim> sHat = 0.5 * (diffU.transpose() + diffU);
        //     real sHatSqr = 2 * sHat.squaredNorm();
        //     real rStar = std::sqrt(sHatSqr) / S;
        //     real DD = 0.5 * (sHatSqr + sqr(S));
        // !    // need second derivatives for rotation term !(CFD++ user manual)
        // }

        real lDES = d;
        // DDES shield
        {
            real lRANS = d;
            real fwStar = .424;
            real nuTur = mufPhy / UMeanXy(0) * std::max((Chi * fnu1), 0.0);
            real nufPhy = mufPhy / UMeanXy(0);
            real rd = (nuTur + nufPhy) / (sqr(kappa) * sqr(d) * std::max(smallReal, diffUNorm));
            real fd = 1. - std::tanh(cube(8 * rd));
            real psiSqr = 0.0;
            if (Chi <= 0)
                psiSqr = 100.0;
            else
                psiSqr = std::min(100.0, (1 - cb1 / (cw1 * sqr(kappa) * fwStar) * (ft2 + (1 - ft2) * fnu2)) /
                                             (fnu1 * std::max(smallReal, 1 - ft2)));
            // if (psiSqr < 1.01)
            // {
            //     std::cout << psiSqr << "Chi " << Chi << " fnu1 " << fnu1 << " xx " << (fnu1 * std::max(smallReal, 1 - ft2)) << " xx " << cb1 / (cw1 * sqr(kappa) * fwStar) << std::endl;
            // }
            real psi = std::sqrt(std::max(psiSqr, 1.0));
            //! note that psi has lower bound of 1
            // psi = 1.0;
            lDES = lRANS - fd * std::max(0., lRANS - lLES * psi);

            // IDDES switch
            if (DESMode == 1 || DESMode == 2)
            {
                const real alphaIDDES = 0.25 - d / hMax;
                const real fB = std::min(2 * std::exp(-9. * sqr(alphaIDDES)), 1.0);

                const real cl = 3.55;
                const real ct = 1.63;
                const real rdt = nuTur / (sqr(kappa) * sqr(d) * std::max(smallReal, diffUNorm));
                const real rdl = nufPhy / (sqr(kappa) * sqr(d) * std::max(smallReal, diffUNorm));
                const real ft = std::tanh(cube(sqr(ct) * rdt));
                const real fl_tmp = sqr(sqr(cl) * rdl);
                const real fl = std::tanh(sqr(sqr(fl_tmp)) * fl_tmp); // power 5

                const real fe1 = 2. * std::exp(-(alphaIDDES >= 0 ? 11.09 : 9.0) * sqr(alphaIDDES));
                const real fe2 = 1. - std::max(ft, fl);
                const real fe = std::max((fe1 - 1.), 0.) * psi * fe2;
                if (DESMode == 2)
                {
                    const real lWMLES = fB * (1 + fe) * lRANS + (1 - fB) * lLES * psi;
                    lDES = lWMLES;
                }
                else
                {
                    const real fdTilde = std::max(fB, std::tanh(cube(8 * rdt)));
                    real lIDDES = fdTilde * (1 + fe) * lRANS + (1 - fdTilde) * lLES * psi;
                    lIDDES = std::max(lLES * smallReal, lIDDES);
                    lDES = lIDDES;
                }
            }
        }
        d = lDES; //! super subs

        const real Sbar = nuh / (sqr(kappa) * sqr(d)) * fnu2;

        real Sh = 0.;

        { // Lee, K., Wilson, M., and Vahdati, M. (April 16, 2018). "Validation of a Numerical Model for Predicting Stalled Flows in a Low-Speed Fan—Part I: Modification of Spalart–Allmaras Turbulence Model." ASME. J. Turbomach. May 2018; 140(5): 051008.
          // real betaSCor = 1;
          // real ch1 = 0.5;
          // real ch2 = 0.7;
          // real a1 = 3; //! is this good?
          // real a2 = 3;
          // Eigen::Vector<real, dim> diffP = (DiffUxy(Seq012, I4) - diffRhoU * velo - UMeanXy(0) * diffU * velo) * (gamma - 1);
          // real veloN = velo.norm();
          // Eigen::Vector<real, dim> uN = velo / (veloN + verySmallReal);
          // real pStar = diffP.dot(uN) / (sqr(UMeanXy(0)) * sqr(veloN) * veloN) * mufPhy;
          // Geom::tPoint omegaV = Geom::CrossMatToVec(Omega);
          // real HStar = omegaV.dot(velo) / (veloN * omegaV.norm() + verySmallReal);
          // real Cs = ch1 * std::tanh(a1 * sqr(pStar)) / std::tanh(1.0) + 1;
          // real Cvh = ch2 * std::tanh(a2 * sqr(HStar)) / std::tanh(1.0) + 1;
          // betaSCor = Cs * Cvh;

            // S *= betaSCor;
        }
#ifdef USE_NS_SA_NEGATIVE_MODEL
        if (Sbar < -cnu2 * S)
            Sh = S + S * (sqr(cnu2) * S + cnu3 * Sbar) / ((cnu3 - 2 * cnu2) * S - Sbar);
        else //*negative fix
#endif
            Sh = S + Sbar;
        // here r is used for fw, we use real d instead of lDES
        const real r = std::min(nuh / (Sh * sqr(kappa * d) + verySmallReal), rlim);
        const real g = r + cw2 * (std::pow(r, 6) - r);
        const real fw = g * std::pow((1 + std::pow(cw3, 6)) / (std::pow(g, 6) + std::pow(cw3, 6)), 1. / 6.);

        // if (d < 0.01)
        //     std::cout << d << " " << lDES << " " << lLES << std::endl;
        // DDES

#ifdef USE_NS_SA_NEGATIVE_MODEL
        real D = (cw1 * fw - cb1 / sqr(kappa) * ft2) * sqr(nuh / lDES); //! modified >>
        real P = cb1 * (1 - ft2) * (Sh + SRotCor) * nuh;                //! modified >>
#else
        real D = (cw1 * fw - cb1 / sqr(kappa) * ft2) * sqr(nuh / lDES);
        real P = cb1 * (1 - ft2) * (Sh + SRotCor) * nuh;
#endif
        real fn = 1;
#ifdef USE_NS_SA_NEGATIVE_MODEL
        if (UMeanXy(I4 + 1) < 0)
        {
            real Chi = UMeanXy(I4 + 1) * muRef / mufPhy;
            fn = (SA::cn1 + std::pow(Chi, 3)) / (SA::cn1 - std::pow(Chi, 3));
            D = -cw1 * sqr(nuh / lDES);
            P = cb1 * (1 - ct3) * std::abs(S + SRotCor) * nuh;
        }
#endif

        if (mode == 0)
            source(I4 + 1) = UMeanXy(0) * (P - D + diffNu.squaredNorm() * cb2 / sigma) / muRef -
                             (UMeanXy(I4 + 1) * fn * muRef + mufPhy) / (UMeanXy(0) * sigma) * diffRho.dot(diffNu) / muRef;
        else
            source(I4 + 1) = SAVersion == 0
                                 ? -std::min(UMeanXy(0) * (std::min(P, 0.0) * 1 - D * 2) / muRef / (std::abs(UMeanXy(I4 + 1)) + verySmallReal), -verySmallReal)
                                 : -std::min(UMeanXy(0) * (P * 0 - D * 2) / muRef / (std::abs(UMeanXy(I4 + 1)) + verySmallReal), -verySmallReal);

        if (!source.allFinite())
        {
            std::cout << P << std::endl;
            std::cout << D << std::endl;
            std::cout << UMeanXy(0) << std::endl;
            std::cout << Sh << std::endl;
            std::cout << nuh << std::endl;
            std::cout << g << std::endl;
            std::cout << r << std::endl;
            std::cout << S << std::endl;
            std::cout << d << std::endl;
            std::cout << fnu2 << std::endl;
            std::cout << mufPhy << std::endl;
            DNDS_assert(false);
        }
    }
}
