/**
 * @file EulerEvaluator.hpp
 * @brief Core finite-volume evaluator for compressible Navier-Stokes / Euler equations.
 *
 * Provides the EulerEvaluator class template, parameterized by EulerModel, which
 * encapsulates the spatial discretization of the compressible Navier-Stokes equations
 * using Compact Finite Volume (CFV) methods with variational reconstruction.
 *
 * Responsibilities include:
 * - Right-hand side (RHS) evaluation of the semi-discrete system
 * - Local time step estimation
 * - Implicit Jacobian assembly (LU-SGS / SGS / GMRES preconditioning)
 * - Boundary condition ghost-value generation for all supported BC types
 * - Wall distance computation (AABB tree, batched AABB, p-Poisson)
 * - Positivity-preserving limiters and increment compression
 * - RANS turbulence model source terms (SA, k-omega SST, k-omega Wilcox, Realizable k-e)
 * - Periodic boundary handling and rotating-frame transformations
 *
 * Supported model specializations (via EulerModel enum):
 *   NS, NS_2D, NS_SA, NS_SA_3D, NS_2EQ, NS_2EQ_3D, NS_3D
 *
 * @see EulerSolver.hpp   Top-level solver orchestrating time marching
 * @see EulerBC.hpp        Boundary condition handler
 * @see Gas.hpp            Gas thermodynamic and Riemann solver routines
 */
#pragma once

// #ifndef __DNDS_REALLY_COMPILING__
// #define __DNDS_REALLY_COMPILING__
// #define __DNDS_REALLY_COMPILING__HEADER_ON__
// #endif

#include "Gas.hpp"
#include "Geom/Mesh.hpp"
#include "CFV/VariationalReconstruction.hpp"
#include "DNDS/JsonUtil.hpp"
#include "Euler.hpp"
#include "EulerBC.hpp"
#include "EulerJacobian.hpp"
#include "EulerEvaluatorSettings.hpp"
#include "DNDS/SerializerBase.hpp"
#include "RANS_ke.hpp"

// #ifdef __DNDS_REALLY_COMPILING__HEADER_ON__
// #undef __DNDS_REALLY_COMPILING__
// #endif

#include "DNDS/JsonUtil.hpp"
#include "fmt/core.h"
#include <iomanip>
#include <functional>

// #define DNDS_FV_EULEREVALUATOR_SOURCE_TERM_ZERO
// // #define DNDS_FV_EULEREVALUATOR_IGNORE_SOURCE_TERM
// // #define DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM

// // #ifdef DNDS_FV_EULEREVALUATOR_IGNORE_SOURCE_TERM // term dependency
// // // #define DNDS_FV_EULEREVALUATOR_USE_SCALAR_JACOBIAN
// // #endif

namespace DNDS::Euler
{

    /**
     * @brief Core finite-volume evaluator for compressible Navier-Stokes / Euler equations.
     *
     * Implements the spatial discretization of the compressible N-S equations on
     * unstructured meshes using Compact Finite Volume (CFV) with variational
     * reconstruction. Handles inviscid and viscous flux evaluation, source terms
     * (including RANS turbulence models), implicit Jacobian assembly, boundary
     * condition ghost-value generation, and positivity-preserving fixes.
     *
     * @tparam model  EulerModel enum selecting the equation set and spatial dimension
     *                (NS, NS_2D, NS_SA, NS_SA_3D, NS_2EQ, NS_2EQ_3D, NS_3D).
     */
    template <EulerModel model = NS>
    class EulerEvaluator
    {
    public:
        using Traits = EulerModelTraits<model>; ///< Compile-time traits: hasSA, has2EQ, etc.
        static const int nVarsFixed = getnVarsFixed(model); ///< Number of conserved variables (compile-time fixed).
        static const int dim = getDim_Fixed(model);         ///< Spatial dimension (2 or 3).
        static const int gDim = getGeomDim_Fixed(model);    ///< Geometric dimension of the mesh.
        static const auto I4 = dim + 1;                     ///< Index of the energy equation (= dim+1).

        static const int MaxBatch = 16; ///< Maximum batch size for vectorized quadrature-point evaluation.
        /// @brief Compute the maximum batch-multiplied column count for batched Eigen matrices.
        static constexpr int MaxBatchMult(int n) { return MaxBatch > 0 ? (n * MaxBatch) : Eigen::Dynamic; }

        typedef Eigen::VectorFMTSafe<real, dim> TVec;           ///< Spatial vector (dim components).
        typedef Eigen::MatrixFMTSafe<real, dim, Eigen::Dynamic, Eigen::ColMajor, dim, MaxBatch> TVec_Batch; ///< Batch of spatial vectors.
        typedef Eigen::MatrixFMTSafe<real, dim, dim> TMat;      ///< Spatial matrix (dim x dim).
        typedef Eigen::MatrixFMTSafe<real, dim, Eigen::Dynamic, Eigen::ColMajor, dim, MaxBatchMult(3)> TMat_Batch; ///< Batch of spatial matrices.
        typedef Eigen::VectorFMTSafe<real, nVarsFixed> TU;      ///< Conservative variable vector (nVarsFixed components).
        typedef Eigen::MatrixFMTSafe<real, nVarsFixed, Eigen::Dynamic, Eigen::ColMajor, nVarsFixed, MaxBatch> TU_Batch; ///< Batch of conservative variable vectors.
        typedef Eigen::MatrixFMTSafe<real, 1, Eigen::Dynamic, Eigen::RowMajor, 1, MaxBatch> TReal_Batch; ///< Batch of scalar values.
        typedef Eigen::MatrixFMTSafe<real, nVarsFixed, nVarsFixed> TJacobianU;  ///< Jacobian matrix (nVars x nVars) of the flux w.r.t. conserved variables.
        typedef Eigen::MatrixFMTSafe<real, dim, nVarsFixed> TDiffU;             ///< Gradient of conserved variables (dim x nVars).
        typedef Eigen::MatrixFMTSafe<real, Eigen::Dynamic, nVarsFixed, Eigen::ColMajor, MaxBatchMult(3)> TDiffU_Batch; ///< Batch of gradient matrices.
        typedef Eigen::MatrixFMTSafe<real, nVarsFixed, dim> TDiffUTransposed;   ///< Transposed gradient (nVars x dim).
        typedef ArrayDOFV<nVarsFixed> TDof;     ///< Cell-centered DOF array (mean values).
        typedef ArrayRECV<nVarsFixed> TRec;     ///< Reconstruction coefficient array (per-cell polynomial coefficients).
        typedef ArrayRECV<1> TScalar;           ///< Scalar reconstruction coefficient array.

        typedef CFV::VariationalReconstruction<gDim> TVFV;      ///< Variational reconstruction type for this geometric dimension.
        typedef ssp<CFV::VariationalReconstruction<gDim>> TpVFV; ///< Shared pointer to the variational reconstruction object.
        typedef ssp<BoundaryHandler<model>> TpBCHandler;         ///< Shared pointer to the boundary condition handler.

    public:
        /**
         * @brief Initialize the finite-volume infrastructure on the mesh.
         *
         * Sets periodic transformations on the VFV object, constructs cell/face
         * geometric metrics, builds reconstruction base functions and weights
         * (with BC-type-dependent Dirichlet/Neumann weighting), and computes
         * reconstruction coefficients.
         *
         * @param mesh       Shared pointer to the unstructured mesh.
         * @param vfv        Shared pointer to the variational reconstruction object.
         * @param pBCHandler Shared pointer to the boundary condition handler (used
         *                   for per-face reconstruction weight selection).
         */
        static void InitializeFV(ssp<Geom::UnstructuredMesh> &mesh, TpVFV vfv, TpBCHandler pBCHandler)
        {
            vfv->SetPeriodicTransformations(
                [mesh](auto u, Geom::t_index id) //! important caveat! using & captures mesh shared_ptr as reference, lost if inbound pointer is destoried!!
                {
                    DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
                    u(EigenAll, Seq123) = mesh->periodicInfo.TransVector<dim, Eigen::Dynamic>(u(EigenAll, Seq123).transpose(), id).transpose();
                },
                [mesh](auto u, Geom::t_index id)
                {
                    DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
                    u(EigenAll, Seq123) = mesh->periodicInfo.TransVectorBack<dim, Eigen::Dynamic>(u(EigenAll, Seq123).transpose(), id).transpose();
                });

            vfv->ConstructMetrics();
            vfv->ConstructBaseAndWeight(
                [&](Geom::t_index id, int iOrder) -> real
                {
                    auto type = pBCHandler->GetTypeFromID(id);
                    if (type == BCSpecial || type == BCOut)
                        return 0;
                    if (type == BCFar) // use Dirichlet type
                        return iOrder ? 0. : 1.;
                    if (type == BCWallInvis || type == BCSym)
                        return iOrder ? 0. : 1.;
                    if (Geom::FaceIDIsPeriodic(id))
                        return iOrder ? 1. : 1.; //! treat as real internal
                    // others: use Dirichlet type
                    return iOrder ? 0. : 1.;
                });
            vfv->ConstructRecCoeff();
        }

    public:
        // static const int gdim = 2; //* geometry dim

    private:
        int nVars = 5; ///< Runtime number of conserved variables (may differ from nVarsFixed for dynamic models).

        bool passiveDiscardSource = false; ///< When true, discard source terms for passive scalar equations.

    public:
        /// @brief Enable or disable discarding of passive-scalar source terms.
        void setPassiveDiscardSource(bool n) { passiveDiscardSource = n; }

    private:
        int axisSymmetric = 0; ///< Non-zero if the solver operates in axisymmetric mode (2D axisymmetric).

    public:
        /// @brief Return the axisymmetric mode flag.
        int GetAxisSymmetric() { return axisSymmetric; }

    private:
    public:
        ssp<Geom::UnstructuredMesh> mesh;                        ///< Shared pointer to the unstructured mesh.
        ssp<CFV::VariationalReconstruction<gDim>> vfv; //! gDim -> 3 for intellisense //!tmptmp  ///< Variational reconstruction object.
        ssp<BoundaryHandler<model>> pBCHandler;                  ///< Boundary condition handler.
        int kAv = 0;                                             ///< Artificial viscosity polynomial order (maxOrder + 1).

        // buffer for fdtau
        // std::vector<real> lambdaCell;
        std::vector<real> lambdaFace;       ///< Per-face spectral radius (inviscid + viscous combined).
        std::vector<real> lambdaFaceC;      ///< Per-face convective spectral radius.
        std::vector<real> lambdaFaceVis;    ///< Per-face viscous spectral radius.
        std::vector<real> lambdaFace0;      ///< Per-face eigenvalue |u·n| (contact wave).
        std::vector<real> lambdaFace123;    ///< Per-face eigenvalue |u·n + a| (acoustic wave).
        std::vector<real> lambdaFace4;      ///< Per-face eigenvalue |u·n - a| (acoustic wave).
        std::vector<real> deltaLambdaFace;  ///< Per-face spectral radius difference for implicit diagonal.
        ArrayDOFV<1> deltaLambdaCell;       ///< Per-cell accumulated spectral radius difference.

        // grad fix
        std::vector<TDiffU> gradUFix; ///< Green-Gauss gradient correction buffer for source term stabilization.

        // wall distance
        std::vector<Eigen::Vector<real, Eigen::Dynamic>> dWall;  ///< Per-cell wall distance (one value per cell node/quadrature point).
        std::vector<real> dWallFace;                             ///< Per-face wall distance (interpolated from cell values).

        // maps from bc id to various objects
        std::map<Geom::t_index, AnchorPointRecorder<nVarsFixed>> anchorRecorders; ///< Per-BC anchor point recorders for profile extraction.
        std::map<Geom::t_index, OneDimProfile<nVarsFixed>> profileRecorders;      ///< Per-BC 1-D profile recorders.
        std::map<Geom::t_index, IntegrationRecorder> bndIntegrations;             ///< Per-BC boundary flux/force integration accumulators.
        std::map<Geom::t_index, std::ofstream> bndIntegrationLogs;                ///< Per-BC log file streams for integration output.

        std::set<Geom::t_index> cLDriverBndIDs;  ///< Boundary IDs driven by the CL (lift-coefficient) driver.
        std::unique_ptr<CLDriver> pCLDriver;     ///< Lift-coefficient driver for AoA adaptation (null if unused).

        // ArrayVDOF<25> dRdUrec;
        // ArrayVDOF<25> dRdb;
        ArrayGRADV<nVarsFixed, gDim> uGradBuf, uGradBufNoLim; ///< Gradient buffers: limited and unlimited.

        Eigen::Vector<real, -1> fluxWallSum;       ///< Accumulated wall flux integral (force on wall boundaries).
        std::vector<TU> fluxBnd;                   ///< Per-boundary-face flux values.
        std::vector<TVec> fluxBndForceT;           ///< Per-boundary-face tangential force.
        index nFaceReducedOrder = 0;               ///< Count of faces where reconstruction order was reduced.

        ssp<Direct::SerialSymLUStructure> symLU;   ///< Symmetric LU structure for direct preconditioner.

        EulerEvaluatorSettings<model> settings;    ///< Physics and numerics settings for this evaluator.

        /**
         * @brief Construct an EulerEvaluator and initialize all internal buffers.
         *
         * Allocates per-face spectral-radius buffers, per-boundary flux arrays,
         * gradient correction arrays (if enabled), wall distance fields (for RANS),
         * and the symmetric LU structure for direct preconditioning. Also validates
         * CL-driver boundary configuration.
         *
         * @param Nmesh       Shared pointer to the unstructured mesh.
         * @param Nvfv        Shared pointer to the variational reconstruction object.
         * @param npBCHandler Shared pointer to the boundary condition handler.
         * @param nSettings   Physics and numerics settings.
         * @param n_nVars     Runtime number of conserved variables (default from model).
         */
        EulerEvaluator(const decltype(mesh) &Nmesh, const decltype(vfv) &Nvfv, const decltype(pBCHandler) &npBCHandler,
                       const EulerEvaluatorSettings<model> &nSettings,
                       int n_nVars = getNVars(model))
            : nVars(n_nVars), axisSymmetric(Nvfv->GetAxisSymmetric()), mesh(Nmesh), vfv(Nvfv), pBCHandler(npBCHandler), kAv(Nvfv->getSettings().maxOrder + 1), settings(nSettings)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            if (getNVars(model) == DynamicSize)
                DNDS_assert_info(nVars >= getDim_Fixed(model) + 2, "nVars too small");
            else
                DNDS_assert_info(nVars == getNVars(model), "do not change the nVars for this model");

            vfv->BuildUGrad(uGradBuf, nVars);
            vfv->BuildUGrad(uGradBufNoLim, nVars);

            if (axisSymmetric)
                DNDS_assert_info(!settings.ignoreSourceTerm, "you have set source term, do not use ignoreSourceTerm! ");

            // lambdaCell.resize(mesh->NumCellProc()); // but only dist part are used, ghost part to not judge for it in facial iter
            lambdaFace.resize(mesh->NumFaceProc());
            lambdaFaceC.resize(mesh->NumFaceProc());
            lambdaFaceVis.resize(lambdaFace.size());
            lambdaFace0.resize(mesh->NumFaceProc(), 0.);
            lambdaFace123.resize(mesh->NumFaceProc(), 0.);
            lambdaFace4.resize(mesh->NumFaceProc(), 0.);

            deltaLambdaFace.resize(lambdaFace.size());
            vfv->BuildUDof(deltaLambdaCell, 1);

            if (settings.useSourceGradFixGG)
            {
                gradUFix.resize(mesh->NumCell());
                for (auto &v : gradUFix)
                    v.resize(Eigen::NoChange, nVars);
            }

            fluxBnd.resize(mesh->NumBnd());
            for (auto &v : fluxBnd)
                v.resize(nVars);
            fluxBndForceT.resize(mesh->NumBnd());

            this->GetWallDist();

            if (model == NS_2EQ || model == NS_2EQ_3D)
            {
                TU farPrim = settings.farFieldStaticValue;
                real gamma = settings.idealGasProperty.gamma;
                Gas::IdealGasThermalConservative2Primitive<dim>(settings.farFieldStaticValue, farPrim, gamma);
                real T = farPrim(I4) / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * farPrim(0));
                // auto [rhs0, rhs] = RANS::SolveZeroGradEquilibrium<dim>(settings.farFieldStaticValue, this->muEff(settings.farFieldStaticValue, T));
                // if(mesh->getMPI().rank == 0)
                //     log()
                //     << "EulerEvaluator===EulerEvaluator: got 2EQ init for farFieldStaticValue: " << settings.farFieldStaticValue.transpose() << "\n"
                //     << fmt::format(" [{:.3e} -> {:.3e}] ", rhs0, rhs) << std::endl;
            }

            symLU = std::make_shared<Direct::SerialSymLUStructure>(mesh->getMPI(), mesh->NumCell());

            for (auto &name : settings.cLDriverBCNames)
            {
                auto bcID = pBCHandler->GetIDFromName(name);
                cLDriverBndIDs.insert(bcID);
                if (bcID >= Geom::BC_ID_DEFAULT_MAX)
                    DNDS_assert_info(pBCHandler->GetFlagFromIDSoft(bcID, "integrationOpt") == 1,
                                     "you have to set integrationOption == 1 to make this bc available by CLDriver");
                else
                    DNDS_assert_info(bcID == Geom::BC_ID_DEFAULT_WALL || bcID == Geom::BC_ID_DEFAULT_WALL_INVIS,
                                     "default bc must be WALL or WALL_INVIS for CLDriver");
            }
            if (cLDriverBndIDs.size())
                pCLDriver = std::make_unique<CLDriver>(settings.cLDriverSettings);
        }

        /**
         * @brief Compute wall distance for all cells (dispatcher).
         *
         * Selects the wall distance algorithm based on the wallDistScheme setting
         * (AABB tree, batched AABB, p-Poisson, or combination) and populates dWall
         * and dWallFace arrays used by RANS turbulence models.
         */
        void GetWallDist();

    private:
        /// Collect wall-boundary triangles for CGAL AABB queries.
        /// @param useQuadPatches  If true, triangulate using quadrature patches (scheme 1);
        ///                        otherwise use element vertices directly (scheme 0/20).
        void GetWallDist_CollectTriangles(bool useQuadPatches,
                                          std::vector<Eigen::Matrix<real, 3, 3>> &trianglesOut);

        /// Wall distance via global CGAL AABB tree (wallDistScheme 0, 1, 20 first pass).
        void GetWallDist_AABB();

        /// Wall distance via batched per-rank CGAL AABB queries (wallDistScheme 2, 20 refine pass).
        void GetWallDist_BatchedAABB();

        /// Wall distance via p-Poisson equation (wallDistScheme 3).
        void GetWallDist_Poisson();

        /// Compute dWallFace from dWall (shared postprocess).
        void GetWallDist_ComputeFaceDistances();

    public:

        /******************************************************/
        static const uint64_t DT_No_Flags = 0x0ull;                 ///< No flags for EvaluateDt.
        static const uint64_t DT_Dont_update_lambda01234 = 0x1ull << 0; ///< Skip recomputation of per-face eigenvalues lambda0/123/4.
        /**
         * @brief Estimate the local or global time step for each cell.
         *
         * Computes the local pseudo-time step dTau based on CFL number, spectral radii
         * of inviscid and viscous fluxes, and optionally enforces a global minimum.
         * Updates per-face eigenvalue arrays (lambda0, lambda123, lambda4) unless
         * DT_Dont_update_lambda01234 is set.
         *
         * @param[out] dt        Per-cell time step array (overwritten).
         * @param[in]  u         Cell-centered conservative variable DOFs.
         * @param[in]  uRec      Reconstruction coefficients.
         * @param[in]  CFL       CFL number.
         * @param[out] dtMinall  Global minimum time step (MPI-reduced).
         * @param[in]  MaxDt     Upper bound on the time step.
         * @param[in]  UseLocaldt If true, use local (per-cell) time stepping.
         * @param[in]  t         Current simulation time.
         * @param[in]  flags     Bitwise combination of DT_* flags.
         */
        void EvaluateDt(
            ArrayDOFV<1> &dt,
            ArrayDOFV<nVarsFixed> &u,
            ArrayRECV<nVarsFixed> &uRec,
            real CFL, real &dtMinall, real MaxDt,
            bool UseLocaldt,
            real t,
            uint64_t flags = DT_No_Flags);

        /// @name RHS evaluation flags (bitwise OR combinable)
        /// @{
        static const uint64_t RHS_No_Flags = 0x0ull;                              ///< Default: full RHS evaluation.
        static const uint64_t RHS_Ignore_Viscosity = 0x1ull << 0;                 ///< Skip viscous flux contribution.
        static const uint64_t RHS_Dont_Update_Integration = 0x1ull << 1;          ///< Skip boundary integration accumulation.
        static const uint64_t RHS_Dont_Record_Bud_Flux = 0x1ull << 2;            ///< Skip recording per-boundary flux.
        static const uint64_t RHS_Direct_2nd_Rec = 0x1ull << 8;                  ///< Use direct 2nd-order reconstruction.
        static const uint64_t RHS_Direct_2nd_Rec_1st_Conv = 0x1ull << 9;         ///< 2nd-order rec with 1st-order convection.
        static const uint64_t RHS_Direct_2nd_Rec_use_limiter = 0x1ull << 10;     ///< Apply limiter when using direct 2nd rec.
        static const uint64_t RHS_Direct_2nd_Rec_already_have_uGradBufNoLim = 0x1ull << 11; ///< uGradBufNoLim is already computed.
        static const uint64_t RHS_Recover_IncFScale = 0x1ull << 12;              ///< Recover incremental face scaling.
        /// @}

        /**
         * @brief Evaluate the spatial right-hand side of the semi-discrete system.
         *
         * Computes inviscid flux (Riemann solver), viscous flux, and source terms
         * over all internal and boundary faces, accumulating cell residuals. Also
         * records boundary force/flux integrations and updates reduced-order face counts.
         *
         * @param[out] rhs            Cell residual array (overwritten).
         * @param[out] JSource        Diagonal Jacobian block from source terms.
         * @param[in]  u              Cell-centered conservative DOFs.
         * @param[in]  uRecUnlim      Unlimited reconstruction coefficients.
         * @param[in]  uRec           Limited reconstruction coefficients.
         * @param[in]  uRecBeta       Per-cell reconstruction compression factor (PP limiter).
         * @param[in]  cellRHSAlpha   Per-cell RHS scaling factor (PP limiter).
         * @param[in]  onlyOnHalfAlpha If true, evaluate only cells with alpha < 1.
         * @param[in]  t              Current simulation time.
         * @param[in]  flags          Bitwise combination of RHS_* flags.
         */
        void EvaluateRHS(
            ArrayDOFV<nVarsFixed> &rhs,
            JacobianDiagBlock<nVarsFixed> &JSource,
            ArrayDOFV<nVarsFixed> &u,
            ArrayRECV<nVarsFixed> &uRecUnlim,
            ArrayRECV<nVarsFixed> &uRec,
            ArrayDOFV<1> &uRecBeta,
            ArrayDOFV<1> &cellRHSAlpha,
            bool onlyOnHalfAlpha,
            real t,
            uint64_t flags = RHS_No_Flags);

        /**
         * @brief Assemble the diagonal blocks of the implicit Jacobian for LU-SGS / SGS.
         *
         * Computes J_diag = (V/dTau + alphaDiag * sum_faces(spectral_radius)) * I + J_source
         * for each cell, where V is cell volume and dTau is the local pseudo-time step.
         *
         * @param[out] JDiag       Per-cell diagonal Jacobian block (overwritten).
         * @param[in]  JSource     Source-term Jacobian contribution to diagonal.
         * @param[in]  dTau        Per-cell local pseudo-time step.
         * @param[in]  dt          Physical time step (for dual time stepping).
         * @param[in]  alphaDiag   Diagonal scaling factor for implicit relaxation.
         * @param[in]  u           Cell-centered conservative DOFs.
         * @param[in]  uRec        Reconstruction coefficients.
         * @param[in]  jacobianCode Controls Jacobian approximation: 0=scalar, 1=analytical flux Jacobian.
         * @param[in]  t           Current simulation time.
         */
        void LUSGSMatrixInit(
            JacobianDiagBlock<nVarsFixed> &JDiag,
            JacobianDiagBlock<nVarsFixed> &JSource,
            ArrayDOFV<1> &dTau, real dt, real alphaDiag,
            ArrayDOFV<nVarsFixed> &u,
            ArrayRECV<nVarsFixed> &uRec,
            int jacobianCode,
            real t);

        /**
         * @brief Compute the matrix-vector product A * uInc for the implicit system.
         *
         * Evaluates the action of the approximate Jacobian on an increment vector,
         * used as the matvec operation inside GMRES.
         *
         * @param[in]  alphaDiag Diagonal scaling factor.
         * @param[in]  t         Current simulation time.
         * @param[in]  u         Cell-centered conservative DOFs.
         * @param[in]  uInc      Increment vector to multiply.
         * @param[in]  JDiag     Pre-assembled diagonal Jacobian blocks.
         * @param[out] AuInc     Result of A * uInc (overwritten).
         */
        void LUSGSMatrixVec(
            real alphaDiag,
            real t,
            ArrayDOFV<nVarsFixed> &u,
            ArrayDOFV<nVarsFixed> &uInc,
            JacobianDiagBlock<nVarsFixed> &JDiag,
            ArrayDOFV<nVarsFixed> &AuInc);

        /**
         * @brief Build the local LU factorization of the Jacobian for direct solve.
         *
         * Assembles the full local Jacobian (including off-diagonal face coupling)
         * and factorizes it, storing the result in jacLU for subsequent direct solves.
         *
         * @param[in]  alphaDiag Diagonal scaling factor.
         * @param[in]  t         Current simulation time.
         * @param[in]  u         Cell-centered conservative DOFs.
         * @param[in]  JDiag     Pre-assembled diagonal Jacobian blocks.
         * @param[out] jacLU     Local LU factorization result (overwritten).
         */
        void LUSGSMatrixToJacobianLU(
            real alphaDiag,
            real t,
            ArrayDOFV<nVarsFixed> &u,
            JacobianDiagBlock<nVarsFixed> &JDiag,
            JacobianLocalLU<nVarsFixed> &jacLU);

        /**
         * @brief Deprecated: use UpdateSGS with uIncIsZero=true instead.
         * Kept for backward compatibility. Delegates to UpdateSGS.
         */
        [[deprecated("Use UpdateSGS with uIncIsZero=true instead")]]
        void UpdateLUSGSForward(
            real alphaDiag,
            real t,
            ArrayDOFV<nVarsFixed> &rhs,
            ArrayDOFV<nVarsFixed> &u,
            ArrayDOFV<nVarsFixed> &uInc,
            JacobianDiagBlock<nVarsFixed> &JDiag,
            ArrayDOFV<nVarsFixed> &uIncNew);

        /**
         * @brief Deprecated: use UpdateSGS with uIncIsZero=true instead.
         * Kept for backward compatibility. Delegates to UpdateSGS.
         */
        [[deprecated("Use UpdateSGS with uIncIsZero=true instead")]]
        void UpdateLUSGSBackward(
            real alphaDiag,
            real t,
            ArrayDOFV<nVarsFixed> &rhs,
            ArrayDOFV<nVarsFixed> &u,
            ArrayDOFV<nVarsFixed> &uInc,
            JacobianDiagBlock<nVarsFixed> &JDiag,
            ArrayDOFV<nVarsFixed> &uIncNew);

        /**
         * @brief Symmetric Gauss-Seidel update for the implicit linear system.
         *
         * @param forward     Scan ascending (true) or descending (false).
         * @param gsUpdate    Use Gauss-Seidel update (read from uIncNew for already-processed cells).
         * @param uIncIsZero  If true, uInc is assumed zero for not-yet-processed cells,
         *                    so their flux contribution is skipped (LUSGS optimisation).
         * @param sumInc      Output: accumulated absolute increment for convergence tracking.
         */
        void UpdateSGS(
            real alphaDiag,
            real t,
            ArrayDOFV<nVarsFixed> &rhs,
            ArrayDOFV<nVarsFixed> &u,
            ArrayDOFV<nVarsFixed> &uInc,
            ArrayDOFV<nVarsFixed> &uIncNew,
            JacobianDiagBlock<nVarsFixed> &JDiag,
            bool forward, bool gsUpdate, TU &sumInc,
            bool uIncIsZero = false);

        /**
         * @brief Solve the implicit linear system using the pre-factored local LU.
         *
         * @param[in]  alphaDiag   Diagonal scaling factor.
         * @param[in]  t           Current simulation time.
         * @param[in]  rhs         Right-hand side residual.
         * @param[in]  u           Cell-centered conservative DOFs.
         * @param[in]  uInc        Current increment (input guess).
         * @param[out] uIncNew     Updated increment (output).
         * @param[out] bBuf        Buffer for intermediate right-hand side assembly.
         * @param[in]  JDiag       Diagonal Jacobian blocks.
         * @param[in]  jacLU       Pre-factored local LU decomposition.
         * @param[in]  uIncIsZero  If true, skip contributions from zero-increment cells.
         * @param[out] sumInc      Accumulated absolute increment for convergence monitoring.
         */
        void LUSGSMatrixSolveJacobianLU(
            real alphaDiag,
            real t,
            ArrayDOFV<nVarsFixed> &rhs,
            ArrayDOFV<nVarsFixed> &u,
            ArrayDOFV<nVarsFixed> &uInc,
            ArrayDOFV<nVarsFixed> &uIncNew,
            ArrayDOFV<nVarsFixed> &bBuf,
            JacobianDiagBlock<nVarsFixed> &JDiag,
            JacobianLocalLU<nVarsFixed> &jacLU,
            bool uIncIsZero,
            TU &sumInc);

        /**
         * @brief SGS sweep coupled with reconstruction update.
         *
         * Performs a single SGS sweep that simultaneously updates the conservative
         * increment (uInc) and the reconstruction increment (uRecInc).
         *
         * @param[in]  alphaDiag Diagonal scaling factor.
         * @param[in]  t         Current simulation time.
         * @param[in]  rhs       Right-hand side residual.
         * @param[in]  u         Cell-centered conservative DOFs.
         * @param[in]  uRec      Reconstruction coefficients.
         * @param[in]  uInc      Current increment for conservative variables.
         * @param[in]  uRecInc   Current reconstruction increment.
         * @param[in]  JDiag     Diagonal Jacobian blocks.
         * @param[in]  forward   Sweep direction: ascending (true) or descending (false).
         * @param[out] sumInc    Accumulated absolute increment for convergence monitoring.
         */
        void UpdateSGSWithRec(
            real alphaDiag,
            real t,
            ArrayDOFV<nVarsFixed> &rhs,
            ArrayDOFV<nVarsFixed> &u,
            ArrayRECV<nVarsFixed> &uRec,
            ArrayDOFV<nVarsFixed> &uInc,
            ArrayRECV<nVarsFixed> &uRecInc,
            JacobianDiagBlock<nVarsFixed> &JDiag,
            bool forward, TU &sumInc);

        // void UpdateLUSGSForwardWithRec(
        //     real alphaDiag,
        //     ArrayDOFV<nVarsFixed> &rhs,
        //     ArrayDOFV<nVarsFixed> &u,
        //     ArrayRECV<nVarsFixed> &uRec,
        //     ArrayDOFV<nVarsFixed> &uInc,
        //     ArrayRECV<nVarsFixed> &uRecInc,
        //     ArrayDOFV<nVarsFixed> &JDiag,
        //     ArrayDOFV<nVarsFixed> &uIncNew);

        /// @brief Clip extreme conserved-variable values to prevent overflow.
        void FixUMaxFilter(ArrayDOFV<nVarsFixed> &u);

        /// @brief Accumulate time-averaged primitive variables for unsteady statistics.
        void TimeAverageAddition(ArrayDOFV<nVarsFixed> &w, ArrayDOFV<nVarsFixed> &wAveraged, real dt, real &tCur);

        /// @brief Convert cell-mean conservative variables to primitive variables.
        void MeanValueCons2Prim(ArrayDOFV<nVarsFixed> &u, ArrayDOFV<nVarsFixed> &w);
        /// @brief Convert cell-mean primitive variables to conservative variables.
        void MeanValuePrim2Cons(ArrayDOFV<nVarsFixed> &w, ArrayDOFV<nVarsFixed> &u);

        using tFCompareField = std::function<TU(const Geom::tPoint &, real)>;         ///< Callback type for analytical comparison field.
        using tFCompareFieldWeight = std::function<real(const Geom::tPoint &, real)>;  ///< Callback type for comparison weighting function.

        /**
         * @brief Compute the norm of the RHS residual vector.
         *
         * @param[out] res     Norm result per variable (resized to nVars).
         * @param[in]  rhs     Cell residual array.
         * @param[in]  P       Norm order (1 = L1, 2 = L2, etc.).
         * @param[in]  volWise If true, weight by cell volume.
         * @param[in]  average If true, divide by total volume/count.
         */
        void EvaluateNorm(Eigen::Vector<real, -1> &res, ArrayDOFV<nVarsFixed> &rhs, index P = 1, bool volWise = false, bool average = false);

        /**
         * @brief Compute the reconstruction error norm (optionally against an analytical field).
         *
         * @param[out] res                 Norm result per variable.
         * @param[in]  u                   Cell-centered conservative DOFs.
         * @param[in]  uRec                Reconstruction coefficients.
         * @param[in]  P                   Norm order.
         * @param[in]  compare             If true, compute error against FCompareField.
         * @param[in]  FCompareField       Analytical field callback.
         * @param[in]  FCompareFieldWeight Weighting callback.
         * @param[in]  t                   Current simulation time.
         */
        void EvaluateRecNorm(
            Eigen::Vector<real, -1> &res,
            ArrayDOFV<nVarsFixed> &u,
            ArrayRECV<nVarsFixed> &uRec,
            index P = 1,
            bool compare = false,
            const tFCompareField &FCompareField = [](const Geom::tPoint &p, real t)
            { return TU::Zero(); },
            const tFCompareFieldWeight &FCompareFieldWeight = [](const Geom::tPoint &p, real t)
            { return 1.0; },
            real t = 0);

        /// @name Limiter flags
        /// @{
        static const uint64_t LIMITER_UGRAD_No_Flags = 0x0ull;                  ///< Default limiter flags.
        static const uint64_t LIMITER_UGRAD_Disable_Shock_Limiter = 0x1ull << 0; ///< Disable shock-detecting component of the limiter.
        /// @}

        /**
         * @brief Apply slope limiter to the gradient field.
         *
         * Limits the reconstructed gradient (uGrad) to produce a monotonicity-preserving
         * gradient (uGradNew). Supports WBAP and CWBAP limiter variants.
         *
         * @param[in]  u         Cell-centered conservative DOFs.
         * @param[in]  uGrad     Input (unlimited) gradient array.
         * @param[out] uGradNew  Output limited gradient array.
         * @param[in]  flags     Bitwise combination of LIMITER_UGRAD_* flags.
         */
        void LimiterUGrad(ArrayDOFV<nVarsFixed> &u, ArrayGRADV<nVarsFixed, gDim> &uGrad, ArrayGRADV<nVarsFixed, gDim> &uGradNew,
                          uint64_t flags = LIMITER_UGRAD_No_Flags);

        static const int EvaluateURecBeta_DEFAULT = 0x00;           ///< Default: evaluate beta without compression.
        static const int EvaluateURecBeta_COMPRESS_TO_MEAN = 0x01; ///< Compress reconstruction toward cell mean to enforce positivity.
        /**
         * @brief Evaluate the positivity-preserving reconstruction limiter (beta).
         *
         * For each cell, computes the maximum compression factor beta such that
         * u_mean + beta * uRec remains physically realizable (positive density and pressure).
         *
         * @param[in]  u          Cell-centered conservative DOFs.
         * @param[in]  uRec       Reconstruction coefficients.
         * @param[out] uRecBeta   Per-cell compression factor in [0,1].
         * @param[out] nLim       Number of cells where beta < 1.
         * @param[out] betaMin    Global minimum beta value.
         * @param[in]  flag       EvaluateURecBeta_DEFAULT or EvaluateURecBeta_COMPRESS_TO_MEAN.
         */
        void EvaluateURecBeta(
            ArrayDOFV<nVarsFixed> &u,
            ArrayRECV<nVarsFixed> &uRec,
            ArrayDOFV<1> &uRecBeta, index &nLim, real &betaMin, int flag);

        /**
         * @brief Assert that all cell-mean values are physically realizable.
         *
         * Checks that density > 0 and internal energy > 0 for all cells.
         *
         * @param[in] u     Cell-centered conservative DOFs.
         * @param[in] panic If true, abort on first violation; otherwise just report.
         * @return true if all cells pass the check.
         */
        bool AssertMeanValuePP(
            ArrayDOFV<nVarsFixed> &u, bool panic);

        static const int EvaluateCellRHSAlpha_DEFAULT = 0x00;          ///< Default alpha evaluation mode.
        static const int EvaluateCellRHSAlpha_MIN_IF_NOT_ONE = 0x01; ///< Take min(alpha, prev) only if prev != 1.
        static const int EvaluateCellRHSAlpha_MIN_ALL = 0x02;        ///< Always take min(alpha, prev).
        /**
         * @brief Compute the positivity-preserving RHS scaling factor (alpha) per cell.
         *
         * Determines the maximum safe scaling alpha in [0,1] such that
         * u + alpha * res remains physically realizable.
         *
         * @param[in]  u              Cell-centered conservative DOFs.
         * @param[in]  uRec           Reconstruction coefficients.
         * @param[in]  uRecBeta       Per-cell reconstruction beta from PP limiter.
         * @param res  Incremental residual (the RHS increment to scale).
         * @param[out] cellRHSAlpha   Per-cell alpha factor.
         * @param[out] nLim           Number of cells where alpha < 1.
         * @param[out] alphaMin       Global minimum alpha.
         * @param[in]  relax          Relaxation factor applied to alpha.
         * @param[in]  compress       Compression mode (1=compress, 0=clamp).
         * @param[in]  flag           EvaluateCellRHSAlpha_* mode flag.
         */
        void EvaluateCellRHSAlpha(
            ArrayDOFV<nVarsFixed> &u,
            ArrayRECV<nVarsFixed> &uRec,
            ArrayDOFV<1> &uRecBeta,
            ArrayDOFV<nVarsFixed> &res,
            ArrayDOFV<1> &cellRHSAlpha, index &nLim, real &alphaMin,
            real relax, int compress = 1,
            int flag = 0);

        /**
         * @brief Expand a previously computed cellRHSAlpha toward 1 where safe.
         *
         * @param res  Incremental residual fixed previously.
         * @param cellRHSAlpha Limiting factor evaluated previously (expanded in-place).
         * @param[out] nLim     Number of cells still limited after expansion.
         * @param[out] alphaMin Global minimum alpha after expansion.
         */
        void EvaluateCellRHSAlphaExpansion(
            ArrayDOFV<nVarsFixed> &u,
            ArrayRECV<nVarsFixed> &uRec,
            ArrayDOFV<1> &uRecBeta,
            ArrayDOFV<nVarsFixed> &res,
            ArrayDOFV<1> &cellRHSAlpha, index &nLim, real alphaMin);

        /// @brief Smooth the local time step across neighboring cells.
        void MinSmoothDTau(
            ArrayDOFV<1> &dTau, ArrayDOFV<1> &dTauNew);

        /******************************************************/

        /**
         * @brief Compute effective molecular viscosity using the configured viscosity model.
         *
         * Supports constant viscosity (muModel=0), Sutherland's law (muModel=1),
         * and density-proportional viscosity (muModel=2).
         *
         * @param U Conservative state vector.
         * @param T Temperature.
         * @return Effective molecular dynamic viscosity.
         */
        real muEff(const TU &U, real T) // TODO: more than sutherland law
        {

            switch (settings.idealGasProperty.muModel)
            {
            case 0:
                return settings.idealGasProperty.muGas;
            case 1:
            {
                real TRel = T / settings.idealGasProperty.TRef;
                return settings.idealGasProperty.muGas *
                       TRel * std::sqrt(TRel) *
                       (settings.idealGasProperty.TRef + settings.idealGasProperty.CSutherland) /
                       (T + settings.idealGasProperty.CSutherland);
            }
            break;
            case 2:
            {
                return settings.idealGasProperty.muGas * U(0);
            }
            break;
            default:
                DNDS_assert_info(false, "No such muModel");
            }
            return std::nan("0");
        }

        /**
         * @brief Compute turbulent eddy viscosity at a face.
         *
         * Dispatches to the appropriate RANS model (SA, k-omega SST, k-omega Wilcox,
         * or Realizable k-epsilon) based on settings.ransModel.
         *
         * @param uMean        Cell-mean conservative state.
         * @param GradUMeanXY  Gradient of conservative variables in physical coordinates.
         * @param muRef        Reference dynamic viscosity scaling.
         * @param muf          Molecular (physical) viscosity at the face.
         * @param iFace        Face index (for wall distance lookup).
         * @return Turbulent eddy viscosity mu_t.
         */
        real getMuTur(const TU &uMean, const TDiffU &GradUMeanXY, real muRef, real muf, index iFace)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            real muTur = 0;
            if constexpr (Traits::hasSA)
            {
                real Chi = uMean(I4 + 1) * muRef / muf;
#ifdef USE_NS_SA_NEGATIVE_MODEL
                if (Chi < 10)
                    Chi = 0.05 * std::log(1 + std::exp(20 * Chi));
#endif
                real Chi3 = cube(Chi);
                real fnu1 = Chi3 / (Chi3 + std::pow(RANS::SA::cnu1, 3));
                muTur = muf * std::max((Chi * fnu1), 0.0);
            }
            if constexpr (Traits::has2EQ)
            {
                real mut = 0;
                if (settings.ransModel == RANSModel::RANS_KOSST)
                    mut = RANS::GetMut_SST<dim>(uMean, GradUMeanXY, muf, dWallFace[iFace]);
                else if (settings.ransModel == RANSModel::RANS_KOWilcox)
                    mut = RANS::GetMut_KOWilcox<dim>(uMean, GradUMeanXY, muf, dWallFace[iFace]);
                else if (settings.ransModel == RANSModel::RANS_RKE)
                    mut = RANS::GetMut_RealizableKe<dim>(uMean, GradUMeanXY, muf, dWallFace[iFace]);
                muTur = mut;
            }
            return muTur;
        }

        /**
         * @brief Compute viscous flux contribution from turbulence model variables.
         *
         * Adds the turbulent diffusion flux for SA (nuTilde) or two-equation (k, omega/epsilon)
         * variables, and the turbulent normal-stress correction to the momentum and energy fluxes.
         *
         * @param UMeanXYC     Cell-mean state in physical coordinates.
         * @param DiffUxyPrimC Gradient of primitive variables in physical coordinates.
         * @param muRef        Reference viscosity scaling.
         * @param mufPhy       Physical molecular viscosity.
         * @param muTur        Turbulent eddy viscosity.
         * @param uNormC       Face outward unit normal.
         * @param iFace        Face index (for wall distance lookup).
         * @param VisFlux      Viscous flux vector (accumulated in-place).
         */
        void visFluxTurVariable(const TU &UMeanXYC, const TDiffU &DiffUxyPrimC,
                                real muRef, real mufPhy, real muTur, const TVec &uNormC, index iFace, TU &VisFlux)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            if constexpr (Traits::hasSA)
            {
                real sigma = RANS::SA::sigma;
                real fn = 1;
#ifdef USE_NS_SA_NEGATIVE_MODEL
                if (UMeanXYC(I4 + 1) < 0)
                {
                    real Chi = UMeanXYC(I4 + 1) * muRef / mufPhy;
                    fn = (RANS::SA::cn1 + std::pow(Chi, 3)) / (RANS::SA::cn1 - std::pow(Chi, 3));
                }
#endif
                VisFlux(I4 + 1) = DiffUxyPrimC(Seq012, {I4 + 1}).dot(uNormC) * (mufPhy + UMeanXYC(I4 + 1) * muRef * fn) / sigma;

                real tauPressure = DiffUxyPrimC(Seq012, Seq123).trace() * (2. / 3.) * (muTur); //! SA's normal stress
                tauPressure *= 0;                                                              // !not standard SA, abandoning
                VisFlux(Seq123) -= tauPressure * uNormC;
                VisFlux(I4) -= tauPressure * UMeanXYC(Seq123).dot(uNormC) / UMeanXYC(0);
            }
            if constexpr (Traits::has2EQ)
            {
                if (settings.ransModel == RANSModel::RANS_KOSST)
                    RANS::GetVisFlux_SST<dim>(UMeanXYC, DiffUxyPrimC, uNormC, muTur, dWallFace[iFace], mufPhy, VisFlux);
                else if (settings.ransModel == RANSModel::RANS_KOWilcox)
                    RANS::GetVisFlux_KOWilcox<dim>(UMeanXYC, DiffUxyPrimC, uNormC, muTur, dWallFace[iFace], mufPhy, VisFlux);
                else if (settings.ransModel == RANSModel::RANS_RKE)
                    RANS::GetVisFlux_RealizableKe<dim>(UMeanXYC, DiffUxyPrimC, uNormC, muTur, dWallFace[iFace], mufPhy, VisFlux);
            }
        }

        /**
         * @brief Transform a conservative state vector from cell frame to face frame for periodic BCs.
         *
         * Applies the periodic rotation/translation to the momentum components when
         * the face is a periodic boundary and the cell is on the donor side.
         *
         * @param[in,out] u     Conservative state vector (modified in-place).
         * @param[in]     iFace Face index.
         * @param[in]     iCell Cell index.
         * @param[in]     if2c  Face-to-cell local index (0=back, 1=front, <0=auto-detect).
         */
        void UFromCell2Face(TU &u, index iFace, index iCell, rowsize if2c)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            if (!mesh->isPeriodic)
                return;
            auto faceID = mesh->GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return;
            if (if2c < 0)
                if2c = vfv->CellIsFaceBack(iCell, iFace) ? 0 : 1;
            if (if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID))
                u(Seq123) = mesh->periodicInfo.TransVectorBack(Eigen::Vector<real, dim>{u(Seq123)}, faceID);
            if (if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID))
                u(Seq123) = mesh->periodicInfo.TransVector(Eigen::Vector<real, dim>{u(Seq123)}, faceID);
        }

        /// @brief Inverse of UFromCell2Face: transform from face frame back to cell frame.
        void UFromFace2Cell(TU &u, index iFace, index iCell, rowsize if2c)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            if (!mesh->isPeriodic)
                return;
            auto faceID = mesh->GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return;
            if (if2c < 0)
                if2c = vfv->CellIsFaceBack(iCell, iFace) ? 0 : 1;
            if (if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID))
                u(Seq123) = mesh->periodicInfo.TransVector(Eigen::Vector<real, dim>{u(Seq123)}, faceID);
            if (if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID))
                u(Seq123) = mesh->periodicInfo.TransVectorBack(Eigen::Vector<real, dim>{u(Seq123)}, faceID);
        }

        /// @brief Transform a state from a neighbor cell across a periodic face.
        void UFromOtherCell(TU &u, index iFace, index iCell, index iCellOther, rowsize if2c)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            auto faceID = mesh->GetFaceZone(iFace);
            if (if2c < 0)
                if2c = vfv->CellIsFaceBack(iCell, iFace) ? 0 : 1;
            mesh->CellOtherCellPeriodicHandle(
                iFace, if2c,
                [&]()
                { u(Seq123) = mesh->periodicInfo.TransVector(Eigen::Vector<real, dim>{u(Seq123)}, faceID); },
                [&]()
                { u(Seq123) = mesh->periodicInfo.TransVectorBack(Eigen::Vector<real, dim>{u(Seq123)}, faceID); });
        }

        /// @brief Transform a gradient tensor from cell frame to face frame for periodic BCs.
        void DiffUFromCell2Face(TDiffU &u, index iFace, index iCell, rowsize if2c, bool reverse = false)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            if (!mesh->isPeriodic)
                return;
            auto faceID = mesh->GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return;
            if (if2c < 0)
                if2c = vfv->CellIsFaceBack(iCell, iFace) ? 0 : 1;
            if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID) && !reverse) ||
                (if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID) && reverse))
            {
                u(Seq012, EigenAll) = mesh->periodicInfo.TransVectorBack<dim, nVarsFixed>(u(Seq012, EigenAll), faceID);
                u(EigenAll, Seq123) = mesh->periodicInfo.TransVectorBack<dim, dim>(u(EigenAll, Seq123).transpose(), faceID).transpose();
            }
            if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID) && !reverse) ||
                (if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID) && reverse))
            {
                u(Seq012, EigenAll) = mesh->periodicInfo.TransVector<dim, nVarsFixed>(u(Seq012, EigenAll), faceID);
                u(EigenAll, Seq123) = mesh->periodicInfo.TransVector<dim, dim>(u(EigenAll, Seq123).transpose(), faceID).transpose();
            }
        }

        /**
         * @brief Compute the numerical flux at a face (batched over quadrature points).
         *
         * Evaluates the Riemann-solver-based inviscid flux and (optionally) the viscous
         * flux at all quadrature points of a face simultaneously. Supports Roe, HLLC,
         * Lax-Friedrichs, and other Riemann solvers selected by rsType.
         *
         * @param[in]  ULxy       Left state at quadrature points (batched).
         * @param[in]  URxy       Right state at quadrature points (batched).
         * @param[in]  ULMeanXy   Left cell mean state.
         * @param[in]  URMeanXy   Right cell mean state.
         * @param[in]  DiffUxy    Gradient of conservative variables at quad points.
         * @param[in]  DiffUxyPrim Gradient of primitive variables at quad points.
         * @param[in]  unitNorm   Face outward unit normals at quad points.
         * @param[in]  vg         Grid velocity at quad points (for ALE / rotating frame).
         * @param[in]  unitNormC  Face-center outward unit normal.
         * @param[in]  vgC        Grid velocity at face center.
         * @param[out] FLfix      Left-biased flux correction (batched).
         * @param[out] FRfix      Right-biased flux correction (batched).
         * @param[out] finc       Numerical flux increment (batched).
         * @param[out] lam0V      Eigenvalue |u·n| at quad points.
         * @param[out] lam123V    Eigenvalue |u·n + a| at quad points.
         * @param[out] lam4V      Eigenvalue |u·n - a| at quad points.
         * @param[in]  btype      Boundary type (UnInitIndex for internal faces).
         * @param[in]  rsType     Riemann solver type (Roe, HLLC, LF, etc.).
         * @param[in]  iFace      Face index.
         * @param[in]  ignoreVis  If true, skip viscous flux computation.
         */
        void fluxFace(
            const TU_Batch &ULxy,
            const TU_Batch &URxy,
            const TU &ULMeanXy,
            const TU &URMeanXy,
            const TDiffU_Batch &DiffUxy,
            const TDiffU_Batch &DiffUxyPrim,
            const TVec_Batch &unitNorm,
            const TVec_Batch &vg,
            const TVec &unitNormC,
            const TVec &vgC,
            TU_Batch &FLfix,
            TU_Batch &FRfix,
            TU_Batch &finc,
            TReal_Batch &lam0V, TReal_Batch &lam123V, TReal_Batch &lam4V,
            Geom::t_index btype,
            typename Gas::RiemannSolverType rsType,
            index iFace, bool ignoreVis);

        /**
         * @brief Compute the source term at a cell quadrature point.
         *
         * Evaluates body force, axisymmetric, rotating-frame, and RANS turbulence
         * model source terms. Optionally computes the source Jacobian for implicit methods.
         *
         * @param[in]  UMeanXy  Conservative state at the quadrature point.
         * @param[in]  DiffUxy  Gradient of conservative variables.
         * @param[in]  pPhy     Physical coordinates of the quadrature point.
         * @param[out] jacobian Source Jacobian matrix (populated if Mode=1 or 2).
         * @param[in]  iCell    Cell index.
         * @param[in]  ig       Quadrature point index within the cell.
         * @param[in]  Mode     0=source vector only, 1=diagonal Jacobian, 2=full Jacobian.
         * @return Source term vector.
         */
        TU source(
            const TU &UMeanXy,
            const TDiffU &DiffUxy,
            const Geom::tPoint &pPhy,
            TJacobianU &jacobian,
            index iCell,
            index ig,
            int Mode) // mode =0: source; mode = 1, diagJacobi; mode = 2,
            ;

        /**
         * @brief inviscid flux approx jacobian (flux term not reconstructed / no riemann)
         *
         */
        auto fluxJacobian0_Right(
            TU &UR,
            const TVec &uNorm,
            Geom::t_index btype)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            DNDS_assert(dim == 3); // only for 3D!!!!!!!!
            const TU &U = UR;
            const TVec &n = uNorm;

            real rhoun = n.dot(U({1, 2, 3}));
            real rhousqr = U({1, 2, 3}).squaredNorm();
            real gamma = settings.idealGasProperty.gamma;
            TJacobianU subFdU;
            subFdU.resize(nVars, nVars);

            subFdU.setZero();
            subFdU(0, 1) = n(1 - 1);
            subFdU(0, 2) = n(2 - 1);
            subFdU(0, 3) = n(3 - 1);
            subFdU(1, 0) = -1.0 / (U(1 - 1) * U(1 - 1)) * U(2 - 1) * rhoun + (1.0 / (U(1 - 1) * U(1 - 1)) * n(1 - 1) * (gamma - 1.0) * (rhousqr - U(1 - 1) * U(5 - 1) * 2.0)) / 2.0 + (U(5 - 1) * n(1 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(1, 1) = (rhoun + U(2 - 1) * n(1 - 1) * 2.0 - U(2 - 1) * gamma * n(1 - 1)) / U(1 - 1);
            subFdU(1, 2) = (U(2 - 1) * n(2 - 1)) / U(1 - 1) - (U(3 - 1) * n(1 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(1, 3) = (U(2 - 1) * n(3 - 1)) / U(1 - 1) - (U(4 - 1) * n(1 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(1, 4) = n(1 - 1) * (gamma - 1.0);
            subFdU(2, 0) = -1.0 / (U(1 - 1) * U(1 - 1)) * U(3 - 1) * rhoun + (1.0 / (U(1 - 1) * U(1 - 1)) * n(2 - 1) * (gamma - 1.0) * (rhousqr - U(1 - 1) * U(5 - 1) * 2.0)) / 2.0 + (U(5 - 1) * n(2 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(2, 1) = (U(3 - 1) * n(1 - 1)) / U(1 - 1) - (U(2 - 1) * n(2 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(2, 2) = (rhoun + U(3 - 1) * n(2 - 1) * 2.0 - U(3 - 1) * gamma * n(2 - 1)) / U(1 - 1);
            subFdU(2, 3) = (U(3 - 1) * n(3 - 1)) / U(1 - 1) - (U(4 - 1) * n(2 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(2, 4) = n(2 - 1) * (gamma - 1.0);
            subFdU(3, 0) = -1.0 / (U(1 - 1) * U(1 - 1)) * U(4 - 1) * rhoun + (1.0 / (U(1 - 1) * U(1 - 1)) * n(3 - 1) * (gamma - 1.0) * (rhousqr - U(1 - 1) * U(5 - 1) * 2.0)) / 2.0 + (U(5 - 1) * n(3 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(3, 1) = (U(4 - 1) * n(1 - 1)) / U(1 - 1) - (U(2 - 1) * n(3 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(3, 2) = (U(4 - 1) * n(2 - 1)) / U(1 - 1) - (U(3 - 1) * n(3 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(3, 3) = (rhoun + U(4 - 1) * n(3 - 1) * 2.0 - U(4 - 1) * gamma * n(3 - 1)) / U(1 - 1);
            subFdU(3, 4) = n(3 - 1) * (gamma - 1.0);
            subFdU(4, 0) = 1.0 / (U(1 - 1) * U(1 - 1) * U(1 - 1)) * rhoun * (-rhousqr + (U(2 - 1) * U(2 - 1)) * gamma + (U(3 - 1) * U(3 - 1)) * gamma + (U(4 - 1) * U(4 - 1)) * gamma - U(1 - 1) * U(5 - 1) * gamma);
            subFdU(4, 1) = 1.0 / (U(1 - 1) * U(1 - 1)) * n(1 - 1) * (-rhousqr + (U(2 - 1) * U(2 - 1)) * gamma + (U(3 - 1) * U(3 - 1)) * gamma + (U(4 - 1) * U(4 - 1)) * gamma - U(1 - 1) * U(5 - 1) * gamma * 2.0) * (-1.0 / 2.0) - 1.0 / (U(1 - 1) * U(1 - 1)) * U(2 - 1) * rhoun * (gamma - 1.0);
            subFdU(4, 2) = 1.0 / (U(1 - 1) * U(1 - 1)) * n(2 - 1) * (-rhousqr + (U(2 - 1) * U(2 - 1)) * gamma + (U(3 - 1) * U(3 - 1)) * gamma + (U(4 - 1) * U(4 - 1)) * gamma - U(1 - 1) * U(5 - 1) * gamma * 2.0) * (-1.0 / 2.0) - 1.0 / (U(1 - 1) * U(1 - 1)) * U(3 - 1) * rhoun * (gamma - 1.0);
            subFdU(4, 3) = 1.0 / (U(1 - 1) * U(1 - 1)) * n(3 - 1) * (-rhousqr + (U(2 - 1) * U(2 - 1)) * gamma + (U(3 - 1) * U(3 - 1)) * gamma + (U(4 - 1) * U(4 - 1)) * gamma - U(1 - 1) * U(5 - 1) * gamma * 2.0) * (-1.0 / 2.0) - 1.0 / (U(1 - 1) * U(1 - 1)) * U(4 - 1) * rhoun * (gamma - 1.0);
            subFdU(4, 4) = (gamma * rhoun) / U(1 - 1);

            real un = rhoun / U(0);

            if constexpr (Traits::hasSA)
            {
                subFdU(5, 5) = un;
                subFdU(5, 0) = -un * U(5) / U(0);
                subFdU(5, 1) = n(0) * U(5) / U(0);
                subFdU(5, 2) = n(1) * U(5) / U(0);
                subFdU(5, 3) = n(2) * U(5) / U(0);
            }
            if constexpr (Traits::has2EQ)
            {
                subFdU(5, 5) = un;
                subFdU(5, 0) = -un * U(5) / U(0);
                subFdU(5, 1) = n(0) * U(5) / U(0);
                subFdU(5, 2) = n(1) * U(5) / U(0);
                subFdU(5, 3) = n(2) * U(5) / U(0);
                subFdU(6, 6) = un;
                subFdU(6, 0) = -un * U(6) / U(0);
                subFdU(6, 1) = n(0) * U(6) / U(0);
                subFdU(6, 2) = n(1) * U(6) / U(0);
                subFdU(6, 3) = n(2) * U(6) / U(0);
            }
            return subFdU;
        }

        /**
         * @brief inviscid flux approx jacobian (flux term not reconstructed / no riemann)
         * if lambdaMain == veryLargeReal, then use lambda0~4 for roe-flux type jacobian
         *
         */
        TU fluxJacobian0_Right_Times_du(
            const TU &U,
            const TU &UOther,
            const TVec &n,
            const TVec &vg,
            Geom::t_index btype,
            const TU &dU,
            real lambdaMain, real lambdaC, real lambdaVis,
            real lambda0, real lambda123, real lambda4,
            int useRoeTerm, int incFsign = -1, int omitF = 0)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            real gamma = settings.idealGasProperty.gamma;
            TVec velo = U(Seq123) / U(0);
            real p, H, asqr;
            Gas::IdealGasThermal(U(I4), U(0), velo.squaredNorm(), gamma, p, asqr, H);
            TVec dVelo;
            real dp;
            Gas::IdealGasUIncrement<dim>(U, dU, velo, gamma, dVelo, dp);
            TU dF(U.size());
            if (omitF == 0)
                Gas::GasInviscidFluxFacialIncrement<dim>(
                    U, dU,
                    n,
                    velo, dVelo, vg,
                    dp, p,
                    dF);
            else
                dF.setZero();
            if (useRoeTerm == 0)
                dF += incFsign * lambdaMain * dU;
            else
            {
                TVec veloRoe;
                real vsqrRoe{0}, aRoe{0}, asqrRoe{0}, HRoe{0};
                TU uMean(U.size());
                Gas::GetRoeAverage<dim>(U, UOther, gamma, veloRoe, vsqrRoe, aRoe, asqrRoe, HRoe, uMean);
                {
                    // TVec dVeloRoe;
                    // real dpRoe;
                    // Gas::IdealGasUIncrement<dim>(uMean, dU, veloRoe, gamma, dVeloRoe, dpRoe);
                    // Gas::GasInviscidFluxFacialIncrement<dim>(
                    //     uMean, dU,
                    //     n,
                    //     veloRoe, dVeloRoe, vg,
                    //     dp, p,
                    //     dF);
                }
                // lambda0 = lambda123 = lambda4 = aRoe + std::abs(veloRoe.dot(n));
                Gas::RoeFluxIncFDiff<dim>(dU, n, veloRoe, vsqrRoe, aRoe, asqrRoe, HRoe,
                                          incFsign * lambda0, incFsign * lambda123, incFsign * lambda4, gamma,
                                          dF);
                dF += incFsign * lambdaVis * dU;
            }
            //! now dF(U, dU) (GasInviscidFluxFacialIncrement) part actually treats the SeqI52Last part (as passive scalars)
            //! the RANS models and eulerEX use the same transport jacobian form

            return dF;
        }

        TJacobianU fluxJacobian0_Right_Times_du_AsMatrix(
            const TU &U,
            const TU &UOther,
            const TVec &n,
            const TVec &vg,
            Geom::t_index btype,
            real lambdaMain, real lambdaC, real lambdaVis,
            real lambda0, real lambda123, real lambda4,
            int useRoeTerm, int incFsign = -1, int omitF = 0)
        { // TODO: optimize this
            TJacobianU J;
            J.resize(nVars, nVars);
            J.setIdentity();
            for (int i = 0; i < nVars; i++)
                J(EigenAll, i) = fluxJacobian0_Right_Times_du(
                    U, UOther, n, vg, btype, J(EigenAll, i),
                    lambdaMain, lambdaC, lambdaVis, lambda0, lambda123, lambda4,
                    useRoeTerm, incFsign, omitF);
            // TODO: for eulerEX, use scalar for SeqI52Last part
            return J;
        }

        TU fluxJacobianC_Right_Times_du(
            const TU &U,
            const TVec &n,
            const TVec &vg,
            Geom::t_index btype,
            const TU &dU)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            real gamma = settings.idealGasProperty.gamma;
            TVec velo = U(Seq123) / U(0);
            real p, H, asqr;
            Gas::IdealGasThermal(U(I4), U(0), velo.squaredNorm(), gamma, p, asqr, H);
            TVec dVelo;
            real dp;
            Gas::IdealGasUIncrement<dim>(U, dU, velo, gamma, dVelo, dp);
            TU dF(U.size());
            Gas::GasInviscidFluxFacialIncrement<dim>(
                U, dU,
                n,
                velo, dVelo, vg,
                dp, p,
                dF);
            return dF;
        }

        /// @brief Get the grid velocity at a face quadrature point (rotating frame).
        TVec GetFaceVGrid(index iFace, index iG)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            TVec ret;
            ret.setZero();
#ifdef USE_ABS_VELO_IN_ROTATION
            if (settings.frameConstRotation.enabled)
                ret += settings.frameConstRotation.vOmega().cross(vfv->GetFaceQuadraturePPhys(iFace, iG) - settings.frameConstRotation.center)(Seq012);
#endif
            return ret;
        }

        /// @brief Get the grid velocity at a face quadrature point (with explicit physical point).
        TVec GetFaceVGrid(index iFace, index iG, const Geom::tPoint &pPhy)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            TVec ret;
            ret.setZero();
#ifdef USE_ABS_VELO_IN_ROTATION
            if (settings.frameConstRotation.enabled)
                ret += settings.frameConstRotation.vOmega().cross(
                    (iG >= -1 ? vfv->GetFaceQuadraturePPhys(iFace, iG) : pPhy) - settings.frameConstRotation.center)(Seq012);
#endif
            return ret;
        }

        /// @brief Get the grid velocity at a face quadrature point from a specific cell's perspective.
        TVec GetFaceVGridFromCell(index iFace, index iCell, int if2c, index iG)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            TVec ret;
            ret.setZero();
#ifdef USE_ABS_VELO_IN_ROTATION
            if (settings.frameConstRotation.enabled)
                ret += settings.frameConstRotation.vOmega().cross(vfv->GetFaceQuadraturePPhysFromCell(iFace, iCell, if2c, iG) - settings.frameConstRotation.center)(Seq012);
#endif
            return ret;
        }

        /// @brief Transform momentum in-place between inertial and rotating frame (velocity only).
        void TransformVelocityRotatingFrame(TU &U, const Geom::tPoint &pPhysics, int direction)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            U(Seq123) += direction * settings.frameConstRotation.vOmega().cross(pPhysics - settings.frameConstRotation.center)(Seq012) * U(0);
        }

        /// @brief Transform full conservative state (momentum + total energy) between frames (relative velocity formulation).
        void TransformURotatingFrame(TU &U, const Geom::tPoint &pPhysics, int direction)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
#ifndef USE_ABS_VELO_IN_ROTATION
            U(I4) -= U(Seq123).squaredNorm() / (2 * U(0));
            U(Seq123) += direction * settings.frameConstRotation.vOmega().cross(pPhysics - settings.frameConstRotation.center)(Seq012) * U(0);
            U(I4) += U(Seq123).squaredNorm() / (2 * U(0));
#endif
        }

        /// @brief Transform full conservative state for the absolute-velocity rotating frame formulation.
        void TransformURotatingFrame_ABS_VELO(TU &U, const Geom::tPoint &pPhysics, int direction)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
#ifdef USE_ABS_VELO_IN_ROTATION
            U(I4) -= U(Seq123).squaredNorm() / (2 * U(0));
            U(Seq123) += direction * settings.frameConstRotation.vOmega().cross(pPhysics - settings.frameConstRotation.center)(Seq012) * U(0);
            U(I4) += U(Seq123).squaredNorm() / (2 * U(0));
#endif
        }

        /// @brief Update boundary anchor point recorders from current solution.
        void updateBCAnchors(ArrayDOFV<nVarsFixed> &u, ArrayRECV<nVarsFixed> &uRec)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            for (Geom::t_index i = Geom::BC_ID_DEFAULT_MAX; i < pBCHandler->size(); i++) // init code, consider adding to ctor
            {
                if (pBCHandler->GetFlagFromIDSoft(i, "anchorOpt") == 0)
                    continue;
                if (!anchorRecorders.count(i))
                    anchorRecorders.emplace(std::make_pair(i, AnchorPointRecorder<nVarsFixed>(mesh->getMPI())));
            }
            for (auto &v : anchorRecorders)
                v.second.Reset();
            for (index iBnd = 0; iBnd < mesh->NumBnd(); iBnd++)
            {
                index iFace = mesh->bnd2faceV.at(iBnd);
                if (iFace < 0) // remember that some iBnd do not have iFace (for periodic case)
                    continue;
                auto f2c = mesh->face2cell[iFace];
                auto gFace = vfv->GetFaceQuad(iFace);

                Geom::Elem::SummationNoOp noOp;
                auto faceBndID = mesh->GetFaceZone(iFace);
                auto faceBCType = pBCHandler->GetTypeFromID(faceBndID);

                if (pBCHandler->GetFlagFromIDSoft(faceBndID, "anchorOpt") == 0)
                    continue;
                gFace.IntegrationSimple(
                    noOp,
                    [&](auto finc, int iG)
                    {
                        TU ULxy = u[f2c[0]];
                        ULxy += (vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, iG, std::array<int, 1>{0}, 1) *
                                 uRec[f2c[0]])
                                    .transpose();
                        this->UFromCell2Face(ULxy, iFace, f2c[0], 0);
                        real dist = vfv->GetFaceQuadraturePPhys(iFace, iG).norm();
                        if (pBCHandler->GetValueExtraFromID(faceBndID).size() >= 3)
                        {
                            Geom::tPoint vOrig = pBCHandler->GetValueExtraFromID(faceBndID)({0, 1, 2});
                            dist = (vfv->GetFaceQuadraturePPhys(iFace, iG) - vOrig).norm();
                        }
                        anchorRecorders.at(faceBndID).AddAnchor(ULxy, dist);
                    });
            }
            for (auto &v : anchorRecorders)
                v.second.ObtainAnchorMPI();
        }

        /// @brief Update boundary 1-D profiles from the current solution.
        void updateBCProfiles(ArrayDOFV<nVarsFixed> &u, ArrayRECV<nVarsFixed> &uRec);

        /// @brief Update boundary profiles using radial-equilibrium pressure extrapolation.
        void updateBCProfilesPressureRadialEq();

        /**
         * @brief Generate the ghost (boundary) state for a boundary face.
         *
         * Dispatches to type-specific BC handlers (far-field, wall, outflow, inflow,
         * symmetry, etc.) based on btype. Used by the RHS evaluator to obtain the
         * right state at boundary faces for the Riemann solver.
         *
         * @param[in,out] ULxy      Left (interior) state; may be modified for certain BCs.
         * @param[in]     ULMeanXy  Left cell-mean state (base state for linearized mode).
         * @param[in]     iCell     Cell index adjacent to the boundary face.
         * @param[in]     iFace     Face index.
         * @param[in]     iG        Quadrature point index (< -1 for arbitrary location).
         * @param[in]     uNorm     Face outward unit normal.
         * @param[in]     normBase  Orthonormal basis with uNorm as first column.
         * @param[in]     pPhysics  Physical coordinates of the evaluation point.
         * @param[in]     t         Current simulation time.
         * @param[in]     btype     Boundary type ID.
         * @param[in]     fixUL     If true, do not modify the left state.
         * @param[in]     geomMode  Geometry evaluation mode (0=standard).
         * @param[in]     linMode   Linearization mode (0=nonlinear, nonzero=linearized about ULMeanXy).
         * @return Ghost (right) state for the Riemann solver.
         */
        TU generateBoundaryValue(
            TU &ULxy, //! warning, possible that UL is also modified
            const TU &ULMeanXy,
            index iCell, index iFace, int iG,
            const TVec &uNorm,
            const TMat &normBase,
            const Geom::tPoint &pPhysics,
            real t,
            Geom::t_index btype,
            bool fixUL = false,
            int geomMode = 0,
            int linMode = 0);

    private:
        /// @name Per-BC-type handlers (called by generateBoundaryValue)
        /// @{

        /// Characteristic-based far-field / outflow-pressure BC (BCFar, BCOutP, BCSpecial far-field).
        TU generateBV_FarField(
            TU &ULxy, const TU &ULMeanXy,
            index iCell, index iFace, int iG,
            const TVec &uNorm, const TMat &normBase,
            const Geom::tPoint &pPhysics, real t,
            Geom::t_index btype, bool fixUL, int geomMode);

        /// Analytical / special far-field BCs (DMR, RT, IV, 2D Riemann, Noh).
        TU generateBV_SpecialFar(
            TU &ULxy, const TU &ULMeanXy,
            index iCell, index iFace, int iG,
            const TVec &uNorm, const TMat &normBase,
            const Geom::tPoint &pPhysics, real t,
            Geom::t_index btype);

        /// Inviscid wall / symmetry BC (BCWallInvis, BCSym).
        TU generateBV_InviscidWall(
            TU &ULxy, const TU &ULMeanXy,
            index iCell, index iFace, int iG,
            const TVec &uNorm, const TMat &normBase,
            const Geom::tPoint &pPhysics, real t,
            Geom::t_index btype);

        /// Viscous no-slip wall BC (BCWall, BCWallIsothermal), including RANS wall treatment.
        TU generateBV_ViscousWall(
            TU &ULxy, const TU &ULMeanXy,
            index iCell, index iFace, int iG,
            const TVec &uNorm, const TMat &normBase,
            const Geom::tPoint &pPhysics, real t,
            Geom::t_index btype, bool fixUL, int linMode);

        /// Supersonic / extrapolation outflow BC (BCOut).
        TU generateBV_Outflow(
            TU &ULxy, const TU &ULMeanXy,
            index iCell, index iFace, int iG,
            const TVec &uNorm, const TMat &normBase,
            const Geom::tPoint &pPhysics, real t,
            Geom::t_index btype);

        /// Prescribed inflow BC (BCIn).
        TU generateBV_Inflow(
            TU &ULxy, const TU &ULMeanXy,
            index iCell, index iFace, int iG,
            const TVec &uNorm, const TMat &normBase,
            const Geom::tPoint &pPhysics, real t,
            Geom::t_index btype);

        /// Total pressure / total temperature inflow BC (BCInPsTs).
        TU generateBV_TotalConditionInflow(
            TU &ULxy, const TU &ULMeanXy,
            index iCell, index iFace, int iG,
            const TVec &uNorm, const TMat &normBase,
            const Geom::tPoint &pPhysics, real t,
            Geom::t_index btype);
        /// @}

    public:

        /// @brief Write boundary profile data to CSV files (rank 0 only).
        void PrintBCProfiles(const std::string &name, ArrayDOFV<nVarsFixed> &u, ArrayRECV<nVarsFixed> &uRec)
        {
            this->updateBCProfiles(u, uRec);
            if (mesh->getMPI().rank != 0)
                return; //! only 0 needs to write
            for (auto &[id, bcProfile] : profileRecorders)
            {
                std::string fname = name + "_bc[" + pBCHandler->GetNameFormID(id) + "]_profile.csv";
                std::filesystem::path outFile{fname};
                std::filesystem::create_directories(outFile.parent_path() / ".");
                std::ofstream fout(fname);
                DNDS_assert_info(fout, fmt::format("failed to open [{}]", fname));
                bcProfile.OutProfileCSV(fout);
            }
        }

        /// @brief Print boundary integration results (force/flux) to console on rank 0.
        void ConsoleOutputBndIntegrations()
        {
            for (auto &i : bndIntegrations)
            {
                auto intOpt = pBCHandler->GetFlagFromIDSoft(i.first, "integrationOpt");
                if (mesh->getMPI().rank == 0)
                {
                    Eigen::VectorFMTSafe<real, Eigen::Dynamic> vPrint = i.second.v;
                    if (intOpt == 2)
                        vPrint(Eigen::seq(nVars, nVars + 1)) /= i.second.div;
                    log() << fmt::format("Bnd [{}] integarted values option [{}] : {:.5e}",
                                         pBCHandler->GetNameFormID(i.first),
                                         intOpt, vPrint.transpose())
                          << std::endl;
                }
            }
        }

        /// @brief Append a line to the per-BC boundary integration CSV log files.
        void BndIntegrationLogWriteLine(const std::string &name, index step, index stage, index iter)
        {
            if (mesh->getMPI().rank != 0)
                return; //! only 0 needs to write
            for (auto &[id, bndInt] : bndIntegrations)
            {
                auto intOpt = pBCHandler->GetFlagFromIDSoft(id, "integrationOpt");
                if (!bndIntegrationLogs.count(id))
                {
                    std::string fname = name + "_bc[" + pBCHandler->GetNameFormID(id) + "]_integrationLog.csv";
                    bndIntegrationLogs.emplace(std::make_pair(id, std::ofstream(fname)));
                    DNDS_assert_info(bndIntegrationLogs.at(id), fmt::format("failed to open [{}]", fname));
                    bndIntegrationLogs.at(id) << "step, stage, iter";
                    for (int i = 0; i < bndInt.v.size(); i++)
                        bndIntegrationLogs.at(id) << ", F" << std::to_string(i);
                    bndIntegrationLogs.at(id) << "\n";
                }
                Eigen::Vector<real, Eigen::Dynamic> vPrint = bndInt.v;
                if (intOpt == 2)
                    vPrint(Eigen::seq(nVars, nVars + 1)) /= bndInt.div;
                bndIntegrationLogs.at(id) << step << ", " << stage << ", " << iter << std::setprecision(16) << std::scientific;
                for (auto &val : vPrint)
                    bndIntegrationLogs.at(id) << ", " << val;
                bndIntegrationLogs.at(id) << std::endl;
            }
        }

        /**
         * @brief Query the CL driver for current lift/drag coefficients and update AoA.
         *
         * Collects boundary force integrals from CL-driven boundaries, projects them
         * onto lift/drag directions, and feeds the result to the CLDriver for AoA adaptation.
         *
         * @param iter Current iteration count.
         * @return Tuple of (CL, CD, AoA) after the driver update.
         */
        std::tuple<real, real, real> CLDriverGetIntegrationUpdate(index iter)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            if (!pCLDriver)
                return {0.0, 0.0, 0.0};
            TU fluxBndCur;
            fluxBndCur.setZero(nVars);
            for (auto bcID : cLDriverBndIDs)
            {
                if (bcID >= Geom::BC_ID_DEFAULT_MAX)
                {
                    fluxBndCur += bndIntegrations.at(bcID).v;
                }
                else
                    fluxBndCur += this->fluxWallSum;
            }
            Geom::tPoint fluxFaceForce;
            fluxFaceForce.setZero();
            fluxFaceForce(Seq012) = fluxBndCur(Seq123);
            fluxFaceForce = pCLDriver->GetAOARotation().transpose() * fluxFaceForce;
            real CLCur = fluxFaceForce.dot(pCLDriver->GetCL0Direction()) * pCLDriver->GetForce2CoeffRatio();
            real CDCur = fluxFaceForce.dot(pCLDriver->GetCD0Direction()) * pCLDriver->GetForce2CoeffRatio();
            pCLDriver->Update(iter, CLCur, mesh->getMPI());
            real AOACur = pCLDriver->GetAOA();
            return {CLCur, CDCur, AOACur};
        }

        /**
         * @brief Compress a reconstruction increment to maintain positivity.
         *
         * Given a cell mean and reconstruction increment, returns umean + uRecInc
         * with the increment clamped so that density, internal energy, and (for RANS)
         * turbulent variables remain non-negative.
         *
         * @param umean     Cell-mean conservative state.
         * @param uRecInc   Reconstruction polynomial increment at an evaluation point.
         * @param compressed Set to true if compression was applied.
         * @return Compressed reconstructed state.
         */
        inline TU CompressRecPart(
            const TU &umean,
            const TU &uRecInc,
            bool &compressed)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            // if (umean(0) + uRecInc(0) < 0)
            // {
            //     std::cout << umean.transpose() << std::endl
            //               << uRecInc.transpose() << std::endl;
            //     DNDS_assert(false);
            // }
            // return umean + uRecInc; // ! no compress shortcut
            // return umean; // ! 0th order shortcut

            // // * Compress Method
            // real compressT = 0.00001;
            // real eFixRatio = 0.00001;
            // Eigen::Vector<real, 5> ret;

            // real compress = 1.0;
            // if ((umean(0) + uRecInc(0)) < umean(0) * compressT)
            //     compress *= umean(0) * (1 - compressT) / uRecInc(0);

            // ret = umean + uRecInc * compress;

            // real Ek = ret({1, 2, 3}).squaredNorm() * 0.5 / (verySmallReal + ret(0));
            // real eT = eFixRatio * Ek;
            // real e = ret(4) - Ek;
            // if (e < 0)
            //     e = eT * 0.5;
            // else if (e < eT)
            //     e = (e * e + eT * eT) / (2 * eT);
            // ret(4) = e + Ek;
            // // * Compress Method

            // TU ret = umean + uRecInc;
            // real eK = ret(Seq123).squaredNorm() * 0.5 / (verySmallReal + std::abs(ret(0)));
            // real e = ret(I4) - eK;
            // if (e <= 0 || ret(0) <= 0)
            //     ret = umean, compressed = true;
            // if constexpr (Traits::hasSA)
            //     if (ret(I4 + 1) < 0)
            //         ret = umean, compressed = true;

            bool rhoFixed = false;
            bool eFixed = false;
            TU ret = umean + uRecInc;
            // do rho fix
            if (ret(0) < 0)
            {
                // rhoFixed = true;
                // TVec veloOld = umean(Seq123) / umean(0);
                // real eOld = umean(I4) - 0.5 * veloOld.squaredNorm() * umean(0);
                // ret(0) = umean(0) * std::exp(uRecInc(0) / (umean(0) + verySmallReal));
                // ret(Seq123) = veloOld * ret(0);
                // ret(I4) = eOld + 0.5 * veloOld.squaredNorm() * ret(0);

                ret = umean;
                compressed = true;
            }

            real eK = ret(Seq123).squaredNorm() * 0.5 / (verySmallReal + ret(0));
            real e = ret(I4) - eK;
            if (e < 0)
            {
                // eFixed = true;
                // real eOld = umean(I4) - eK;
                // real eNew = eOld * std::exp(eOld / (umean(I4) - eK));
                // ret(I4) = eNew + eK;

                ret = umean;

                compressed = true;
            }

#ifdef USE_NS_SA_NUT_REDUCED_ORDER
            if constexpr (Traits::hasSA)
                if (ret(I4 + 1) < 0)
                    ret(I4 + 1) = umean(I4 + 1), compressed = true;
#endif
            if constexpr (Traits::has2EQ)
            {
                if (ret(I4 + 1) < 0)
                    ret(I4 + 1) = umean(I4 + 1), compressed = true;
                if (ret(I4 + 2) < 0)
                    ret(I4 + 2) = umean(I4 + 2), compressed = true;
            }

            return ret;
        }

        /**
         * @brief Compress a solution increment to maintain positivity.
         *
         * Given the current state u and an update increment uInc, returns a modified
         * increment that ensures u + result has positive density, positive internal
         * energy, and (for RANS) non-negative turbulent variables. Uses exponential
         * decay clamping for density and a quadratic solve for internal energy.
         *
         * @param u    Current conservative state.
         * @param uInc Proposed increment.
         * @return Modified (compressed) increment safe to add to u.
         */
        inline TU CompressInc(
            const TU &u,
            const TU &uInc)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            real rhoEps = smallReal * settings.refUPrim(0) * 1e-1;
            real pEps = smallReal * settings.refUPrim(I4) * 1e-1;
            if (settings.ppEpsIsRelaxed)
            {
                pEps *= verySmallReal, rhoEps *= verySmallReal;
            }

            TU ret = uInc;

            /** A intuitive fix **/ //! need positive perserving technique!
            DNDS_assert(u(0) > 0);
            if (u(0) + ret(0) <= rhoEps)
            {
                real declineV = ret(0) / (u(0) + verySmallReal);
                real newrho = u(0) * std::exp(declineV);
                // newrho = std::max(newrho, rhoEps);
                newrho = rhoEps;
                ret *= (newrho - u(0)) / (ret(0) - verySmallReal);
                // std::cout << (newrho - u(0)) / (ret(0) + verySmallReal) << std::endl;
                // DNDS_assert(false);
            }
            real ekOld = 0.5 * u(Seq123).squaredNorm() / (u(0) + verySmallReal);
            real rhoEinternal = u(I4) - ekOld;
            DNDS_assert(rhoEinternal > 0);
            real ek = 0.5 * (u(Seq123) + ret(Seq123)).squaredNorm() / (u(0) + ret(0) + verySmallReal);
            real rhoEinternalNew = u(I4) + ret(I4) - ek;
            if (rhoEinternalNew <= pEps)
            {
                real declineV = (rhoEinternalNew - rhoEinternal) / (rhoEinternal + verySmallReal);
                real newrhoEinteralNew = (std::exp(declineV) + verySmallReal) * rhoEinternal;
                real gamma = settings.idealGasProperty.gamma;
                // newrhoEinteralNew = std::max(pEps / (gamma - 1), newrhoEinteralNew);
                newrhoEinteralNew = pEps / (gamma - 1);
                real c0 = 2 * u(I4) * u(0) - u(Seq123).squaredNorm() - 2 * u(0) * newrhoEinteralNew;
                real c1 = 2 * u(I4) * ret(0) + 2 * u(0) * ret(I4) - 2 * u(Seq123).dot(ret(Seq123)) - 2 * ret(0) * newrhoEinteralNew;
                real c2 = 2 * ret(I4) * ret(0) - ret(Seq123).squaredNorm();
                real deltaC = sqr(c1) - 4 * c0 * c2;
                DNDS_assert(deltaC > 0);
                real alphaL = (-std::sqrt(deltaC) - c1) / (2 * c2);
                real alphaR = (std::sqrt(deltaC) - c1) / (2 * c2);
                // if (c2 > 0)
                //     DNDS_assert(alphaL > 0);
                // DNDS_assert(alphaR > 0);
                // DNDS_assert(alphaL < 1);
                // if (c2 < 0)
                //     DNDS_assert(alphaR < 1);
                real alpha = std::min((c2 > 0 ? alphaL : alphaL), 1.);
                alpha = std::max(0., alpha);
                ret *= alpha * (0.99);

                real decay = 1 - 1e-1;
                for (int iter = 0; iter < 1000; iter++)
                {
                    ek = 0.5 * (u(Seq123) + ret(Seq123)).squaredNorm() / (u(0) + ret(0) + verySmallReal);
                    if (ret(I4) + u(I4) - ek < newrhoEinteralNew)
                        ret *= decay, alpha *= decay;
                    else
                        break;
                }

                ek = 0.5 * (u(Seq123) + ret(Seq123)).squaredNorm() / (u(0) + ret(0) + verySmallReal);

                if (ret(I4) + u(I4) - ek < newrhoEinteralNew * 0.5)
                {
                    std::cout << std::scientific << std::setprecision(5);
                    std::cout << u(0) << " " << ret(0) << std::endl;
                    std::cout << rhoEinternalNew << " " << rhoEinternal << std::endl;
                    std::cout << declineV << std::endl;
                    std::cout << newrhoEinteralNew << std::endl;
                    std::cout << ret(I4) + u(I4) - ek << std::endl;
                    std::cout << alpha << std::endl;
                    DNDS_assert(false);
                }
            }

            /** A intuitive fix **/
// #define USE_NS_SA_ALLOW_NEGATIVE_MEAN
#ifndef USE_NS_SA_ALLOW_NEGATIVE_MEAN
            if constexpr (Traits::hasSA)
            {
                if (u(I4 + 1) + ret(I4 + 1) < 0)
                {
                    // std::cout << "Fixing SA inc " << std::endl;

                    DNDS_assert(u(I4 + 1) >= 0); //! might be bad using gmres, add this to gmres inc!
                    real declineV = ret(I4 + 1) / (u(I4 + 1) + 1e-6);
                    real newu5 = u(I4 + 1) * std::exp(declineV);
                    // ! refvalue:
                    real muRef = settings.idealGasProperty.muGas;
                    newu5 = std::max(1e-6, newu5);
                    ret(I4 + 1) = newu5 - u(I4 + 1);
                }
            }
#endif
            if constexpr (Traits::has2EQ)
            {
                if (u(I4 + 1) + ret(I4 + 1) < 0)
                {
                    // std::cout << "Fixing KE inc " << std::endl;

                    DNDS_assert(u(I4 + 1) >= 0); //! might be bad using gmres, add this to gmres inc!
                    real declineV = ret(I4 + 1) / (u(I4 + 1) + 1e-6);
                    real newu5 = u(I4 + 1) * std::exp(declineV);
                    // ! refvalue:
                    real muRef = settings.idealGasProperty.muGas;
                    // newu5 = std::max(1e-10, newu5);
                    ret(I4 + 1) = newu5 - u(I4 + 1);
                }

                if (u(I4 + 2) + ret(I4 + 2) < 0)
                {
                    // std::cout << "Fixing KE inc " << std::endl;

                    DNDS_assert(u(I4 + 2) >= 0); //! might be bad using gmres, add this to gmres inc!
                    real declineV = ret(I4 + 2) / (u(I4 + 2) + 1e-6);
                    real newu5 = u(I4 + 2) * std::exp(declineV);
                    // ! refvalue:
                    real muRef = settings.idealGasProperty.muGas;
                    // newu5 = std::max(1e-10, newu5);
                    ret(I4 + 2) = newu5 - u(I4 + 2);
                }
            }

            return ret;
        }

        /// @brief Apply CompressInc to every cell, modifying cxInc in-place.
        void FixIncrement(
            ArrayDOFV<nVarsFixed> &cx,
            ArrayDOFV<nVarsFixed> &cxInc, real alpha = 1.0)
        {
            for (index iCell = 0; iCell < cxInc.Size(); iCell++)
                cxInc[iCell] = this->CompressInc(cx[iCell], cxInc[iCell] * alpha);
        }

        /**
         * @brief Add a positivity-compressed increment to the solution.
         *
         * For each cell, compresses the increment via CompressInc, then adds it to cx.
         * Reports the global minimum compression factor via MPI reduction.
         *
         * @param[in,out] cx     Solution array (updated in-place).
         * @param[in]     cxInc  Increment array.
         * @param[in]     alpha  Scaling factor applied to the increment before compression.
         */
        void AddFixedIncrement(
            ArrayDOFV<nVarsFixed> &cx,
            ArrayDOFV<nVarsFixed> &cxInc, real alpha = 1.0)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            real alpha_fix_min = 1.0;
            for (index iCell = 0; iCell < cxInc.Size(); iCell++)
            {
                TU compressedInc = this->CompressInc(cx[iCell], cxInc[iCell] * alpha);
                real newAlpha = std::abs(compressedInc(0)) /
                                (std::abs((cxInc[iCell] * alpha)(0)));
                if (std::abs((cxInc[iCell] * alpha)(0)) < verySmallReal)
                    newAlpha = 1.; //! old inc could be zero, so compresion alpha is always 1
                alpha_fix_min = std::min(
                    alpha_fix_min,
                    newAlpha);
                // if (newAlpha < 1.0 - 1e-14)
                //     std::cout << "KL\n"
                //               << std::scientific << std::setprecision(5)
                //               << this->CompressInc(cx[iCell], cxInc[iCell] * alpha).transpose() << "\n"
                //               << cxInc[iCell].transpose() * alpha << std::endl;
                cx[iCell] += compressedInc;
                // wall fix in not needed
                // if (model == NS_2EQ || model == NS_2EQ_3D)
                //     if (iCell < mesh->NumCell())
                //         for (auto f : mesh->cell2face[iCell])
                //             if (pBCHandler->GetTypeFromID(mesh->GetFaceZone(f)) == BCWall || pBCHandler->GetTypeFromID(mesh->GetFaceZone(f)) == BCWallIsothermal)
                //             { // for SST or KOWilcox
                //                 TVec uNorm = vfv->GetFaceNorm(f, -1)(Seq012);
                //                 real vt = (cx[iCell](Seq123) - cx[iCell](Seq123).dot(uNorm) * uNorm).norm() / cx[iCell](0);
                //                 // cx[iCell](I4 + 1) = sqr(vt) * cx[iCell](0) * 1; // k = v_tang ^2 in sublayer, Wilcox book
                //                 // cx[iCell](I4 + 1) *= 0;

                //                 real d1 = dWall[iCell].mean();
                //                 // cx[iCell](I4 + 1) = 0.; // superfix, actually works
                //                 // real d1 = dWall[iCell].minCoeff();
                //                 real pMean, asqrMean, Hmean;
                //                 real gamma = settings.idealGasProperty.gamma;
                //                 auto ULMeanXy = cx[iCell];
                //                 Gas::IdealGasThermal(ULMeanXy(I4), ULMeanXy(0), (ULMeanXy(Seq123) / ULMeanXy(0)).squaredNorm(),
                //                                      gamma, pMean, asqrMean, Hmean);
                //                 // ! refvalue:
                //                 real muRef = settings.idealGasProperty.muGas;
                //                 real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * ULMeanXy(0));
                //                 real mufPhy1 = muEff(ULMeanXy, T);

                //                 real rhoOmegaaaWall = mufPhy1 / sqr(d1) * 800;
                //                 // cx[iCell](I4 + 2) = rhoOmegaaaWall * 0.5; // this is bad
                //             }

                if (model == NS_2EQ || model == NS_2EQ_3D)
                { // for SST or KOWilcox
                    if (settings.ransModel == RANSModel::RANS_KOSST ||
                        settings.ransModel == RANSModel::RANS_KOWilcox)
                        cx[iCell](I4 + 2) = std::max(cx[iCell](I4 + 2), settings.RANSBottomLimit * settings.farFieldStaticValue(I4 + 2));
                }
            }
            real alpha_fix_min_c = alpha_fix_min;
            MPI::Allreduce(&alpha_fix_min_c, &alpha_fix_min, 1, DNDS_MPI_REAL, MPI_MIN, cx.father->getMPI().comm);
            if (alpha_fix_min < 1.0)
                if (cx.father->getMPI().rank == 0)
                    log() << TermColor::Magenta << "Increment fixed " << std::scientific << std::setprecision(5) << alpha_fix_min << TermColor::Reset << std::endl;
        }

        // void AddFixedIncrement(
        //     ArrayDOFV<nVarsFixed> &cx,
        //     ArrayDOFV<nVarsFixed> &cxInc, real alpha = 1.0)
        // {
        //     real alpha_fix_min = 1.0;
        //     for (index iCell = 0; iCell < cxInc.Size(); iCell++)
        //     {
        //         TU compressedInc = this->CompressInc(cx[iCell], cxInc[iCell] * alpha);
        //         real newAlpha = std::abs(compressedInc(0)) /
        //                         (std::abs((cxInc[iCell] * alpha)(0)));
        //         if (std::abs((cxInc[iCell] * alpha)(0)) < verySmallReal)
        //             newAlpha = 1.; //! old inc could be zero, so compresion alpha is always 1
        //         alpha_fix_min = std::min(
        //             alpha_fix_min,
        //             newAlpha);
        //         // if (newAlpha < 1.0 - 1e-14)
        //         //     std::cout << "KL\n"
        //         //               << std::scientific << std::setprecision(5)
        //         //               << this->CompressInc(cx[iCell], cxInc[iCell] * alpha).transpose() << "\n"
        //         //               << cxInc[iCell].transpose() * alpha << std::endl;
        //         // cx[iCell] += compressedInc;
        //     }
        //     real alpha_fix_min_c = alpha_fix_min;
        //     MPI::Allreduce(&alpha_fix_min_c, &alpha_fix_min, 1, DNDS_MPI_REAL, MPI_MIN, cx.father->getMPI().comm);
        //     if (alpha_fix_min < 1.0)
        //         if (cx.father->getMPI().rank == 0)
        //             std::cout << "Increment fixed " << std::scientific << std::setprecision(5) << alpha_fix_min << std::endl;

        //     // for (index iCell = 0; iCell < cxInc.Size(); iCell++)
        //     //     cx[iCell] += alpha_fix_min * alpha * cxInc[iCell];
        //     index nFixed = 0;
        //     for (index iCell = 0; iCell < cxInc.Size(); iCell++)
        //     {
        //         TU compressedInc = this->CompressInc(cx[iCell], cxInc[iCell] * alpha);
        //         real newAlpha = std::abs(compressedInc(0)) /
        //                         (std::abs((cxInc[iCell] * alpha)(0)));
        //         if (std::abs((cxInc[iCell] * alpha)(0)) < verySmallReal)
        //             newAlpha = 1.; //! old inc could be zero, so compresion alpha is always 1

        //         // if (newAlpha < 1.0 - 1e-14)
        //         //     std::cout << "KL\n"
        //         //               << std::scientific << std::setprecision(5)
        //         //               << this->CompressInc(cx[iCell], cxInc[iCell] * alpha).transpose() << "\n"
        //         //               << cxInc[iCell].transpose() * alpha << std::endl;
        //         cx[iCell] += (newAlpha < 1 ? nFixed ++,alpha_fix_min : 1) * alpha * cxInc[iCell];
        //     }
        //     index nFixed_c = nFixed;
        //     MPI::Allreduce(&nFixed_c, &nFixed, 1, DNDS_MPI_INDEX, MPI_SUM, cx.father->getMPI().comm);
        //     if (alpha_fix_min < 1.0)
        //         if (cx.father->getMPI().rank == 0)
        //             std::cout << "Increment fixed number " << nFixed_c << std::endl;
        // }

        /**
         * @brief Apply Laplacian smoothing to a residual field.
         *
         * Iteratively smooths r into rs using weighted neighbor averaging.
         * Used to stabilize central-difference residuals on coarse grids.
         *
         * @param[in]     r      Input residual.
         * @param[in,out] rs     Smoothed residual (output; also used as work buffer).
         * @param[out]    rtemp  Temporary buffer (same size as r).
         * @param[in]     nStep  Number of smoothing passes (0 = use settings.nCentralSmoothStep).
         */
        void CentralSmoothResidual(ArrayDOFV<nVarsFixed> &r, ArrayDOFV<nVarsFixed> &rs, ArrayDOFV<nVarsFixed> &rtemp, int nStep = 0)
        {
            for (int iterS = 1; iterS <= (nStep > 0 ? nStep : settings.nCentralSmoothStep); iterS++)
            {
                real epsC = settings.centralSmoothEps;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    real div = 1.;
                    TU vC = r[iCell];
                    auto c2f = mesh->cell2face[iCell];
                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                    {
                        index iFace = c2f[ic2f];
                        index iCellOther = vfv->CellFaceOther(iCell, iFace);
                        if (iCellOther != UnInitIndex)
                        {
                            div += epsC;
                            vC += epsC * rs[iCellOther];
                        }
                    }
                    rtemp[iCell] = vC / div;
                }
                rs = rtemp;
                rs.trans.startPersistentPull();
                rs.trans.waitPersistentPull();
            }
        }

        /**
         * @brief Set initial conservative DOF values for all cells.
         *
         * Populates u with the far-field state (or a problem-specific initial condition)
         * based on the evaluator settings. Handles rotating-frame velocity transformations.
         *
         * @param[out] u Cell-centered DOF array to initialize.
         */
        void InitializeUDOF(ArrayDOFV<nVarsFixed> &u);

        /**
         * @brief References to arrays needed by the output data picker.
         *
         * Groups the solution, reconstruction, and PP-limiter arrays for
         * use by InitializeOutputPicker to set up probe/output callbacks.
         */
        struct OutputOverlapDataRefs
        {
            ArrayDOFV<nVarsFixed> &u;      ///< Cell-centered conservative DOFs.
            ArrayRECV<nVarsFixed> &uRec;   ///< Reconstruction coefficients.
            ArrayDOFV<1> &betaPP;          ///< PP reconstruction limiter beta.
            ArrayDOFV<1> &alphaPP;         ///< PP RHS limiter alpha.
        };

        /// @brief Initialize an OutputPicker with field callbacks for VTK/HDF5 output.
        void InitializeOutputPicker(OutputPicker &op, OutputOverlapDataRefs dataRefs);
    };
}

#define DNDS_EulerEvaluator_INS_EXTERN(model, ext)                                                                        \
    namespace DNDS::Euler                                                                                                 \
    {                                                                                                                     \
        ext template void EulerEvaluator<model>::LUSGSMatrixInit(                                                         \
            JacobianDiagBlock<nVarsFixed> &JDiag,                                                                         \
            JacobianDiagBlock<nVarsFixed> &JSource,                                                                       \
            ArrayDOFV<1> &dTau, real dt, real alphaDiag,                                                                  \
            ArrayDOFV<nVarsFixed> &u,                                                                                     \
            ArrayRECV<nVarsFixed> &uRec,                                                                                  \
            int jacobianCode,                                                                                             \
            real t);                                                                                                      \
                                                                                                                          \
        ext template void EulerEvaluator<model>::LUSGSMatrixVec(                                                          \
            real alphaDiag,                                                                                               \
            real t,                                                                                                       \
            ArrayDOFV<nVarsFixed> &u,                                                                                     \
            ArrayDOFV<nVarsFixed> &uInc,                                                                                  \
            JacobianDiagBlock<nVarsFixed> &JDiag,                                                                         \
            ArrayDOFV<nVarsFixed> &AuInc);                                                                                \
                                                                                                                          \
        ext template void EulerEvaluator<model>::LUSGSMatrixToJacobianLU(                                                 \
            real alphaDiag,                                                                                               \
            real t,                                                                                                       \
            ArrayDOFV<nVarsFixed> &u,                                                                                     \
            JacobianDiagBlock<nVarsFixed> &JDiag,                                                                         \
            JacobianLocalLU<nVarsFixed> &jacLU);                                                                          \
                                                                                                                          \
        ext template void EulerEvaluator<model>::UpdateLUSGSForward(                                                      \
            real alphaDiag,                                                                                               \
            real t,                                                                                                       \
            ArrayDOFV<nVarsFixed> &rhs,                                                                                   \
            ArrayDOFV<nVarsFixed> &u,                                                                                     \
            ArrayDOFV<nVarsFixed> &uInc,                                                                                  \
            JacobianDiagBlock<nVarsFixed> &JDiag,                                                                         \
            ArrayDOFV<nVarsFixed> &uIncNew);                                                                              \
                                                                                                                          \
        ext template void EulerEvaluator<model>::UpdateLUSGSBackward(                                                     \
            real alphaDiag,                                                                                               \
            real t,                                                                                                       \
            ArrayDOFV<nVarsFixed> &rhs,                                                                                   \
            ArrayDOFV<nVarsFixed> &u,                                                                                     \
            ArrayDOFV<nVarsFixed> &uInc,                                                                                  \
            JacobianDiagBlock<nVarsFixed> &JDiag,                                                                         \
            ArrayDOFV<nVarsFixed> &uIncNew);                                                                              \
                                                                                                                          \
        ext template void EulerEvaluator<model>::UpdateSGS(                                                               \
            real alphaDiag,                                                                                               \
            real t,                                                                                                       \
            ArrayDOFV<nVarsFixed> &rhs,                                                                                   \
            ArrayDOFV<nVarsFixed> &u,                                                                                     \
            ArrayDOFV<nVarsFixed> &uInc,                                                                                  \
            ArrayDOFV<nVarsFixed> &uIncNew,                                                                               \
            JacobianDiagBlock<nVarsFixed> &JDiag,                                                                         \
            bool forward, bool gsUpdate, TU &sumInc,                                                                      \
            bool uIncIsZero);                                                                     \
        ext template void EulerEvaluator<model>::UpdateSGSWithRec(                                                        \
            real alphaDiag,                                                                                               \
            real t,                                                                                                       \
            ArrayDOFV<nVarsFixed> &rhs,                                                                                   \
            ArrayDOFV<nVarsFixed> &u,                                                                                     \
            ArrayRECV<nVarsFixed> &uRec,                                                                                  \
            ArrayDOFV<nVarsFixed> &uInc,                                                                                  \
            ArrayRECV<nVarsFixed> &uRecInc,                                                                               \
            JacobianDiagBlock<nVarsFixed> &JDiag,                                                                         \
            bool forward, TU &sumInc);                                                                                    \
                                                                                                                          \
        ext template void EulerEvaluator<model>::LUSGSMatrixSolveJacobianLU(                                              \
            real alphaDiag,                                                                                               \
            real t,                                                                                                       \
            ArrayDOFV<nVarsFixed> &rhs,                                                                                   \
            ArrayDOFV<nVarsFixed> &u,                                                                                     \
            ArrayDOFV<nVarsFixed> &uInc,                                                                                  \
            ArrayDOFV<nVarsFixed> &uIncNew,                                                                               \
            ArrayDOFV<nVarsFixed> &bBuf,                                                                                  \
            JacobianDiagBlock<nVarsFixed> &JDiag,                                                                         \
            JacobianLocalLU<nVarsFixed> &jacLU,                                                                           \
            bool uIncIsZero,                                                                                              \
            TU &sumInc);                                                                                                  \
                                                                                                                          \
        ext template void EulerEvaluator<model>::InitializeUDOF(ArrayDOFV<nVarsFixed> &u);                                \
                                                                                                                          \
        ext template void EulerEvaluator<model>::FixUMaxFilter(                                                           \
            ArrayDOFV<nVarsFixed> &u);                                                                                    \
                                                                                                                          \
        ext template void EulerEvaluator<model>::TimeAverageAddition(                                                     \
            ArrayDOFV<nVarsFixed> &w, ArrayDOFV<nVarsFixed> &wAveraged, real dt, real &tCur);                             \
        ext template void EulerEvaluator<model>::MeanValueCons2Prim(                                                      \
            ArrayDOFV<nVarsFixed> &u, ArrayDOFV<nVarsFixed> &w);                                                          \
        ext template void EulerEvaluator<model>::MeanValuePrim2Cons(                                                      \
            ArrayDOFV<nVarsFixed> &w, ArrayDOFV<nVarsFixed> &u);                                                          \
                                                                                                                          \
        ext template void EulerEvaluator<model>::EvaluateNorm(                                                            \
            Eigen::Vector<real, -1> &res, ArrayDOFV<nVarsFixed> &rhs, index P, bool volWise, bool average);               \
                                                                                                                          \
        ext template void EulerEvaluator<model>::EvaluateRecNorm(                                                         \
            Eigen::Vector<real, -1> &res,                                                                                 \
            ArrayDOFV<nVarsFixed> &u,                                                                                     \
            ArrayRECV<nVarsFixed> &uRec,                                                                                  \
            index P,                                                                                                      \
            bool compare,                                                                                                 \
            const tFCompareField &FCompareField,                                                                          \
            const tFCompareFieldWeight &FCompareFieldWeight,                                                              \
            real t);                                                                                                      \
                                                                                                                          \
        ext template void EulerEvaluator<model>::LimiterUGrad(                                                            \
            ArrayDOFV<nVarsFixed> &u, ArrayGRADV<nVarsFixed, gDim> &uGrad, ArrayGRADV<nVarsFixed, gDim> &uGradNew,        \
            uint64_t flags);                                                                                              \
                                                                                                                          \
        ext template void EulerEvaluator<model>::EvaluateURecBeta(                                                        \
            ArrayDOFV<nVarsFixed> &u,                                                                                     \
            ArrayRECV<nVarsFixed> &uRec,                                                                                  \
            ArrayDOFV<1> &uRecBeta, index &nLim, real &betaMin, int flag);                                                \
                                                                                                                          \
        ext template bool EulerEvaluator<model>::AssertMeanValuePP(                                                       \
            ArrayDOFV<nVarsFixed> &u, bool panic);                                                                        \
                                                                                                                          \
        ext template void EulerEvaluator<model>::EvaluateCellRHSAlpha(                                                    \
            ArrayDOFV<nVarsFixed> &u,                                                                                     \
            ArrayRECV<nVarsFixed> &uRec,                                                                                  \
            ArrayDOFV<1> &uRecBeta,                                                                                       \
            ArrayDOFV<nVarsFixed> &rhs,                                                                                   \
            ArrayDOFV<1> &cellRHSAlpha, index &nLim, real &alphaMin, real relax,                                          \
            int compress,                                                                                                 \
            int flag);                                                                                                    \
                                                                                                                          \
        ext template void EulerEvaluator<model>::EvaluateCellRHSAlphaExpansion(                                           \
            ArrayDOFV<nVarsFixed> &u,                                                                                     \
            ArrayRECV<nVarsFixed> &uRec,                                                                                  \
            ArrayDOFV<1> &uRecBeta,                                                                                       \
            ArrayDOFV<nVarsFixed> &res,                                                                                   \
            ArrayDOFV<1> &cellRHSAlpha, index &nLim, real alphaMin);                                                      \
        ext template void EulerEvaluator<model>::MinSmoothDTau(                                                           \
            ArrayDOFV<1> &dTau, ArrayDOFV<1> &dTauNew);                                                                   \
        ext template void EulerEvaluator<model>::updateBCProfiles(ArrayDOFV<nVarsFixed> &u, ArrayRECV<nVarsFixed> &uRec); \
        ext template void EulerEvaluator<model>::updateBCProfilesPressureRadialEq();                                      \
    }

DNDS_EulerEvaluator_INS_EXTERN(NS, extern);
DNDS_EulerEvaluator_INS_EXTERN(NS_2D, extern);
DNDS_EulerEvaluator_INS_EXTERN(NS_SA, extern);
DNDS_EulerEvaluator_INS_EXTERN(NS_2EQ, extern);
DNDS_EulerEvaluator_INS_EXTERN(NS_3D, extern);
DNDS_EulerEvaluator_INS_EXTERN(NS_SA_3D, extern);
DNDS_EulerEvaluator_INS_EXTERN(NS_2EQ_3D, extern);

#define DNDS_EulerEvaluator_EvaluateDt_INS_EXTERN(model, ext)                                                              \
    namespace DNDS::Euler                                                                                                  \
    {                                                                                                                      \
        ext template void EulerEvaluator<model>::GetWallDist();                                                            \
        ext template void EulerEvaluator<model>::GetWallDist_CollectTriangles(                                             \
            bool, std::vector<Eigen::Matrix<real, 3, 3>> &);                                                               \
        ext template void EulerEvaluator<model>::GetWallDist_AABB();                                                       \
        ext template void EulerEvaluator<model>::GetWallDist_BatchedAABB();                                                \
        ext template void EulerEvaluator<model>::GetWallDist_Poisson();                                                    \
        ext template void EulerEvaluator<model>::GetWallDist_ComputeFaceDistances();                                       \
        ext template void EulerEvaluator<model>::EvaluateDt(                                                               \
            ArrayDOFV<1> &dt,                                                                                              \
            ArrayDOFV<nVarsFixed> &u,                                                                                      \
            ArrayRECV<nVarsFixed> &uRec,                                                                                   \
            real CFL, real &dtMinall, real MaxDt,                                                                          \
            bool UseLocaldt,                                                                                               \
            real t,                                                                                                        \
            uint64_t flags);                                                                                               \
        ext template void                                                                                                  \
        EulerEvaluator<model>::fluxFace(                                                                                   \
            const TU_Batch &ULxy,                                                                                          \
            const TU_Batch &URxy,                                                                                          \
            const TU &ULMeanXy,                                                                                            \
            const TU &URMeanXy,                                                                                            \
            const TDiffU_Batch &DiffUxy,                                                                                   \
            const TDiffU_Batch &DiffUxyPrim,                                                                               \
            const TVec_Batch &unitNorm,                                                                                    \
            const TVec_Batch &vg,                                                                                          \
            const TVec &unitNormC,                                                                                         \
            const TVec &vgC,                                                                                               \
            TU_Batch &FLfix,                                                                                               \
            TU_Batch &FRfix,                                                                                               \
            TU_Batch &finc,                                                                                                \
            TReal_Batch &lam0V, TReal_Batch &lam123V, TReal_Batch &lam4V,                                                  \
            Geom::t_index btype,                                                                                           \
            typename Gas::RiemannSolverType rsType,                                                                        \
            index iFace, bool ignoreVis);                                                                                  \
        ext template                                                                                                       \
            typename EulerEvaluator<model>::TU                                                                             \
            EulerEvaluator<model>::source(                                                                                 \
                const TU &UMeanXy,                                                                                         \
                const TDiffU &DiffUxy,                                                                                     \
                const Geom::tPoint &pPhy,                                                                                  \
                TJacobianU &jacobian,                                                                                      \
                index iCell,                                                                                               \
                index ig,                                                                                                  \
                int Mode);                                                                                                 \
        ext template                                                                                                       \
            typename EulerEvaluator<model>::TU                                                                             \
            EulerEvaluator<model>::generateBoundaryValue(                                                                  \
                TU &ULxy,                                                                                                  \
                const TU &ULMeanXy,                                                                                        \
                index iCell, index iFace, int iG,                                                                          \
                const TVec &uNorm,                                                                                         \
                const TMat &normBase,                                                                                      \
                const Geom::tPoint &pPhysics,                                                                              \
                real t,                                                                                                    \
                Geom::t_index btype,                                                                                       \
                bool fixUL,                                                                                                \
                int geomMode, int linMode);                                                                                \
        ext template typename EulerEvaluator<model>::TU EulerEvaluator<model>::generateBV_FarField(                        \
            TU &, const TU &, index, index, int, const TVec &, const TMat &,                                              \
            const Geom::tPoint &, real, Geom::t_index, bool, int);                                                        \
        ext template typename EulerEvaluator<model>::TU EulerEvaluator<model>::generateBV_SpecialFar(                      \
            TU &, const TU &, index, index, int, const TVec &, const TMat &,                                              \
            const Geom::tPoint &, real, Geom::t_index);                                                                   \
        ext template typename EulerEvaluator<model>::TU EulerEvaluator<model>::generateBV_InviscidWall(                    \
            TU &, const TU &, index, index, int, const TVec &, const TMat &,                                              \
            const Geom::tPoint &, real, Geom::t_index);                                                                   \
        ext template typename EulerEvaluator<model>::TU EulerEvaluator<model>::generateBV_ViscousWall(                     \
            TU &, const TU &, index, index, int, const TVec &, const TMat &,                                              \
            const Geom::tPoint &, real, Geom::t_index, bool, int);                                                        \
        ext template typename EulerEvaluator<model>::TU EulerEvaluator<model>::generateBV_Outflow(                         \
            TU &, const TU &, index, index, int, const TVec &, const TMat &,                                              \
            const Geom::tPoint &, real, Geom::t_index);                                                                   \
        ext template typename EulerEvaluator<model>::TU EulerEvaluator<model>::generateBV_Inflow(                          \
            TU &, const TU &, index, index, int, const TVec &, const TMat &,                                              \
            const Geom::tPoint &, real, Geom::t_index);                                                                   \
        ext template typename EulerEvaluator<model>::TU EulerEvaluator<model>::generateBV_TotalConditionInflow(            \
            TU &, const TU &, index, index, int, const TVec &, const TMat &,                                              \
            const Geom::tPoint &, real, Geom::t_index);                                                                   \
        ext template void EulerEvaluator<model>::InitializeOutputPicker(OutputPicker &op, OutputOverlapDataRefs dataRefs); \
    }

DNDS_EulerEvaluator_EvaluateDt_INS_EXTERN(NS, extern);
DNDS_EulerEvaluator_EvaluateDt_INS_EXTERN(NS_2D, extern);
DNDS_EulerEvaluator_EvaluateDt_INS_EXTERN(NS_SA, extern);
DNDS_EulerEvaluator_EvaluateDt_INS_EXTERN(NS_2EQ, extern);
DNDS_EulerEvaluator_EvaluateDt_INS_EXTERN(NS_3D, extern);
DNDS_EulerEvaluator_EvaluateDt_INS_EXTERN(NS_SA_3D, extern);
DNDS_EulerEvaluator_EvaluateDt_INS_EXTERN(NS_2EQ_3D, extern);

#define DNDS_EulerEvaluator_EvaluateRHS_INS_EXTERN(model, ext) \
    namespace DNDS::Euler                                      \
    {                                                          \
        ext template void EulerEvaluator<model>::EvaluateRHS(  \
            ArrayDOFV<nVarsFixed> &rhs,                        \
            JacobianDiagBlock<nVarsFixed> &JSource,            \
            ArrayDOFV<nVarsFixed> &u,                          \
            ArrayRECV<nVarsFixed> &uRecUnlim,                  \
            ArrayRECV<nVarsFixed> &uRec,                       \
            ArrayDOFV<1> &uRecBeta,                            \
            ArrayDOFV<1> &cellRHSAlpha,                        \
            bool onlyOnHalfAlpha,                              \
            real t,                                            \
            uint64_t flags);                                   \
    }

DNDS_EulerEvaluator_EvaluateRHS_INS_EXTERN(NS, extern);
DNDS_EulerEvaluator_EvaluateRHS_INS_EXTERN(NS_2D, extern);
DNDS_EulerEvaluator_EvaluateRHS_INS_EXTERN(NS_SA, extern);
DNDS_EulerEvaluator_EvaluateRHS_INS_EXTERN(NS_2EQ, extern);
DNDS_EulerEvaluator_EvaluateRHS_INS_EXTERN(NS_3D, extern);
DNDS_EulerEvaluator_EvaluateRHS_INS_EXTERN(NS_SA_3D, extern);
DNDS_EulerEvaluator_EvaluateRHS_INS_EXTERN(NS_2EQ_3D, extern);