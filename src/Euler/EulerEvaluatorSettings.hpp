/** @file EulerEvaluatorSettings.hpp
 *  @brief Complete solver configuration for the Euler/Navier-Stokes evaluator.
 *
 *  Contains the EulerEvaluatorSettings template struct which aggregates all
 *  runtime parameters for the compressible flow solver:
 *  - Jacobian options (scalar vs. Roe, wall treatment).
 *  - Reconstruction and limiting parameters.
 *  - Riemann solver selection and tuning.
 *  - Wall-distance computation settings.
 *  - RANS turbulence model selection (SA, k-omega) and DES length scales.
 *  - Viscous flux and source term options.
 *  - Rotating reference frame (FrameConstRotation).
 *  - CL (lift-coefficient) driver configuration.
 *  - Region-based initial conditions (box, plane, exprtk).
 *  - Ideal gas thermodynamic properties.
 *
 *  All settings use DNDS_DECLARE_CONFIG for automatic JSON schema generation
 *  and serialization/deserialization.
 */
#pragma once
#include "DNDS/JsonUtil.hpp"
#include "DNDS/ConfigParam.hpp"

#include "Euler.hpp"
#include "Gas.hpp"
#include "CLDriver.hpp"
#include <unordered_set>

namespace DNDS::Euler
{
    /**
     * @brief Master configuration struct for the compressible Euler/Navier-Stokes evaluator.
     *
     * Organizes all solver-tunable parameters into a single struct that supports
     * JSON round-trip serialization via DNDS_DECLARE_CONFIG. After deserialization,
     * the finalize() hook validates cross-field constraints (e.g. mutually exclusive
     * Jacobian modes) and computes derived reference quantities.
     *
     * @tparam model  The EulerModel tag (e.g. NS_SA_3D) that determines variable
     *                count, spatial dimension, and available turbulence model traits.
     */
    template <EulerModel model>
    struct EulerEvaluatorSettings
    {
        using Traits = EulerModelTraits<model>;              ///< Compile-time model traits.
        static const int nVarsFixed = getnVarsFixed(model);  ///< Compile-time variable count.
        static const int dim = getDim_Fixed(model);          ///< Physical dimension (2 or 3).
        static const int gDim = getGeomDim_Fixed(model);     ///< Geometric dimension (may differ for axi-symmetric).
        static const auto I4 = dim + 1;                      ///< Index of the energy equation in the state vector.

        /// @name Jacobian Options
        /// @{
        bool useScalarJacobian = false;  ///< Use scalar (diagonal) Jacobian approximation instead of block.
        bool useRoeJacobian = false;     ///< Use Roe-linearization-based Jacobian.
        bool noRsOnWall = false;         ///< Disable the Riemann solver on wall boundary faces.
        bool noGRPOnWall = false;        ///< Disable the Generalized Riemann Problem (GRP) on wall faces.
        bool ignoreSourceTerm = false;   ///< Completely ignore source terms (must be false when RANS or body forces are active).
        /// @}

        /// @name Reconstruction
        /// @{
        int direct2ndRecMethod = 1;        ///< Direct 2nd-order reconstruction method selector.
        int specialBuiltinInitializer = 0; ///< Index of a built-in special initializer (0 = none).
        real uRecAlphaCompressPower = 1;   ///< Alpha compression power for the reconstruction limiter.
        real uRecBetaCompressPower = 1;    ///< Beta compression power for the reconstruction limiter.
        bool forceVolURecBeta = true;      ///< Force volume-based beta in the reconstruction.
        bool ppEpsIsRelaxed = false;       ///< Use relaxed positivity-preserving epsilon.
        /// @}

        real RANSBottomLimit = 0.01;       ///< Lower clamp for RANS turbulence variables.

        /// @name Riemann Solver Configuration
        /// @{
        Gas::RiemannSolverType rsType = Gas::Roe;             ///< Primary Riemann solver type.
        Gas::RiemannSolverType rsTypeAux = Gas::UnknownRS;    ///< Auxiliary Riemann solver type (UnknownRS = same as primary).
        Gas::RiemannSolverType rsTypeWall = Gas::UnknownRS;   ///< Wall-face Riemann solver type (UnknownRS = same as primary).
        real rsFixScale = 1;          ///< Entropy-fix scaling factor for the Riemann solver.
        real rsIncFScale = 1;         ///< Incremental flux scaling factor.
        int rsMeanValueEig = 0;       ///< Mean-value eigenvalue computation mode.
        int rsRotateScheme = 0;       ///< Riemann solver rotation scheme selector.
        /// @}

        /// @name Wall-Distance Computation
        /// @{
        real minWallDist = 1e-12;           ///< Minimum wall distance clamp (avoids singularities).
        int wallDistExection = 0;           ///< Execution mode: 0 = parallel, 1 = serial.
        real wallDistRefineMax = 1;         ///< Maximum wall-distance refinement factor.
        int wallDistScheme = 0;             ///< Wall-distance computation scheme selector.
        int wallDistCellLoadSize = 1024 * 32; ///< Cell batch size for wall-distance computation.
        int wallDistIter = 1000;            ///< Maximum iterations for the wall-distance solver.
        int wallDistLinSolver = 0;          ///< Linear solver: 0 = Jacobi, 1 = GMRES.
        real wallDistResTol = 1e-4;         ///< Residual tolerance for wall-distance convergence.
        int wallDistIterStart = 100;        ///< Starting iteration count for the wall-distance solver.
        int wallDistPoissonP = 2;           ///< Poisson equation power in the wall-distance PDE.
        real wallDistDTauScale = 100.;      ///< Pseudo-time step scaling for wall-distance solver.
        int wallDistNJacobiSweep = 10;      ///< Number of Jacobi sweeps per wall-distance iteration.
        /// @}

        /// @name RANS / DES Configuration
        /// @{
        real SADESScale = veryLargeReal;    ///< SA-DES length scale (veryLargeReal effectively disables DES).
        int SADESMode = 1;                  ///< SA-DES mode selector (1 = DDES, etc.).
        /**
         * @brief SA model variant selector.
         *
         * - **0 (default):** Current formulation — rotation correction uses cRot = 2.0
         *   with corrected strain-rate magnitude |S| = ||S_ij + S_ij^T|| / sqrt(2),
         *   SRotCor = cRot * min(0, |Omega| - |S|), and the implicit Jacobian source
         *   includes the negative part of production: min(P, 0) * 1.
         * - **1 (legacy):** Pre-31578ce (dev/harry_ba3) formulation — rotation correction
         *   uses cRot = 1.0 with the Frobenius norm SS = ||S_ij + S_ij^T||,
         *   SRotCor = cRot * min(0, SS - S), and the implicit Jacobian source omits
         *   production entirely (P * 0).
         */
        int SAVersion = 0;
        RANSModel ransModel = RANSModel::RANS_None; ///< RANS turbulence model (RANS_None, RANS_SA, RANS_KOWilcox, etc.).
        int ransUseQCR = 0;                ///< Enable QCR (Quadratic Constitutive Relation) correction.
        int ransSARotCorrection = 1;       ///< SA rotation/curvature correction mode.
        int ransEigScheme = 0;             ///< Eigenvalue computation scheme for RANS.
        int ransForce2nd = 0;              ///< Force 2nd-order accuracy for RANS variables.
        int ransSource2nd = 0;             ///< Enable 2nd-order RANS source term discretization.
        /// @}

        /// @name Viscous Flux and Source Options
        /// @{
        int source2nd = 0;                 ///< Enable 2nd-order source term discretization.
        int usePrimGradInVisFlux = 0;      ///< Use primitive-variable gradients in viscous flux.
        int useSourceGradFixGG = 0;        ///< Apply Green-Gauss gradient fix for source terms.
        int nCentralSmoothStep = 0;        ///< Number of central-difference smoothing steps.
        real centralSmoothEps = 0.5;       ///< Epsilon for central smoothing.
        Eigen::Vector<real, 3> constMassForce = Eigen::Vector<real, 3>{0, 0, 0}; ///< Constant body force vector [fx, fy, fz].
        /// @}
        /**
         * @brief Constant-rotation reference frame settings.
         *
         * When enabled, the solver transforms the governing equations into
         * a non-inertial frame rotating at a constant angular velocity about
         * the specified axis through the specified center.
         */
        struct FrameConstRotation
        {
            bool enabled = false;                            ///< Enable the rotating frame.
            Geom::tPoint axis = Geom::tPoint{0, 0, 1};      ///< Rotation axis (unit vector; normalized in finalize()).
            Geom::tPoint center = Geom::tPoint{0, 0, 0};    ///< Center of rotation [x, y, z].
            real rpm = 0;                                    ///< Rotational speed in revolutions per minute.

            /// @brief Compute angular velocity magnitude (rad/s) from RPM.
            /// @return Omega = rpm * 2π / 60.
            real Omega()
            {
                return rpm * (2 * pi / 60.);
            }

            /// @brief Return the angular velocity vector (axis * Omega).
            /// @return 3-D omega vector aligned with the rotation axis.
            Geom::tPoint vOmega()
            {
                return axis * Omega();
            }

            /**
             * @brief Project a position vector onto the plane perpendicular to the axis.
             * @param r  Position vector in the absolute frame.
             * @return Radial component of @p r (axis-normal projection).
             */
            Geom::tPoint rVec(const Geom::tPoint &r)
            {
                return r - r.dot(axis) * axis;
            }

            /**
             * @brief Build the local cylindrical (r, θ, z) coordinate frame at position @p r.
             *
             * Column 0 = radial unit vector, column 1 = tangential (axis × r̂),
             * column 2 = axial (same as the rotation axis).
             *
             * @param r  Position vector in the absolute frame.
             * @return 3×3 matrix whose columns are the (r, θ, z) basis vectors.
             */
            Geom::tGPoint rtzFrame(const Geom::tPoint &r)
            {
                Geom::tPoint rn = rVec(r).normalized();
                Geom::tGPoint ret;
                ret(EigenAll, 0) = rn;
                ret(EigenAll, 2) = axis;
                ret(EigenAll, 1) = axis.cross(rn);
                return ret;
            }
            DNDS_DECLARE_CONFIG(FrameConstRotation)
            {
                DNDS_FIELD(enabled, "Enable constant-rotation reference frame");
                DNDS_FIELD(axis,    "Rotation axis (unit vector)");
                DNDS_FIELD(center,  "Rotation center coordinates");
                DNDS_FIELD(rpm,     "Rotational speed in RPM");
            }
        } frameConstRotation; ///< Rotating reference frame configuration.
        CLDriverSettings cLDriverSettings;        ///< Lift-coefficient (CL) driver settings.
        std::vector<std::string> cLDriverBCNames; ///< Boundary zone names for CL driver force integration.
        Eigen::Vector<real, -1> farFieldStaticValue = Eigen::Vector<real, 5>{1, 0, 0, 0, 2.5}; ///< Far-field reference state vector (size = nVars).
        /**
         * @brief Axis-aligned box region for initial condition specification.
         *
         * Cells whose centroids lie within [x0,x1]×[y0,y1]×[z0,z1] are
         * initialized to the state vector @c v.
         */
        struct BoxInitializer
        {
            real x0{0}, x1{0}, y0{0}, y1{0}, z0{0}, z1{0}; ///< Box bounds [min, max] per axis.
            Eigen::Vector<real, -1> v;                       ///< Initial state vector (size = nVars).

            DNDS_DECLARE_CONFIG(BoxInitializer)
            {
                DNDS_FIELD(x0, "Box x-min");
                DNDS_FIELD(x1, "Box x-max");
                DNDS_FIELD(y0, "Box y-min");
                DNDS_FIELD(y1, "Box y-max");
                DNDS_FIELD(z0, "Box z-min");
                DNDS_FIELD(z1, "Box z-max");
                DNDS_FIELD(v,  "Initial value vector (size = nVars)");
            }
        };
        std::vector<BoxInitializer> boxInitializers; ///< List of box-region initial condition specifiers.

        /**
         * @brief Half-space region for initial condition specification.
         *
         * Cells satisfying `a*x + b*y + c*z >= h` are initialized to the
         * state vector @c v. The normal direction is (a, b, c).
         */
        struct PlaneInitializer
        {
            real a{0}, b{0}, c{0}, h{0}; ///< Plane equation coefficients: a*x + b*y + c*z = h.
            Eigen::Vector<real, -1> v;    ///< Initial state vector (size = nVars).

            DNDS_DECLARE_CONFIG(PlaneInitializer)
            {
                DNDS_FIELD(a, "Plane normal x-component");
                DNDS_FIELD(b, "Plane normal y-component");
                DNDS_FIELD(c, "Plane normal z-component");
                DNDS_FIELD(h, "Plane offset");
                DNDS_FIELD(v, "Initial value vector (size = nVars)");
            }
        };
        std::vector<PlaneInitializer> planeInitializers; ///< List of plane-region initial condition specifiers.

        /**
         * @brief Expression-based initial condition using the ExprTk library.
         *
         * Evaluates user-supplied mathematical expressions (one per line in @c exprs)
         * to compute the initial state at each cell centroid. Lines are concatenated
         * with newlines to form a single ExprTk program string.
         */
        struct ExprtkInitializer
        {
            std::vector<std::string> exprs; ///< ExprTk expression lines (concatenated with newlines).

            DNDS_DECLARE_CONFIG(ExprtkInitializer)
            {
                DNDS_FIELD(exprs, "Expression lines (concatenated with newlines)");
            }

            /**
             * @brief Concatenate all expression lines into a single ExprTk program string.
             * @return The full expression string with newline separators.
             */
            std::string GetExpr() const
            {
                std::string ret;
                for (auto &line : exprs)
                    ret += line + "\n";
                return ret;
            }
        };
        std::vector<ExprtkInitializer> exprtkInitializers; ///< List of ExprTk-based initial condition specifiers.

        /**
         * @brief Ideal gas thermodynamic property set.
         *
         * Stores gamma, gas constant, viscosity parameters, and Prandtl number.
         * The heat capacity CpGas is a derived quantity recomputed automatically
         * after deserialization via the post_read hook calling recomputeDerived().
         */
        struct IdealGasProperty
        {
            real gamma = 1.4;                                  ///< Ratio of specific heats (Cp/Cv).
            real Rgas = 1;                                     ///< Specific gas constant (J/(kg·K) in dimensional runs).
            real muGas = 1;                                    ///< Dynamic viscosity (or reference viscosity for Sutherland).
            real prGas = 0.72;                                 ///< Prandtl number.
            real CpGas = Rgas * gamma / (gamma - 1);           ///< Heat capacity at constant pressure (derived, not serialized).
            real TRef = 273.15;                                ///< Reference temperature (K) for Sutherland's law.
            real CSutherland = 110.4;                          ///< Sutherland constant (K).
            int muModel = 1;                                   ///< Viscosity model: 0 = constant, 1 = Sutherland, 2 = constant_nu.

            DNDS_DECLARE_CONFIG(IdealGasProperty)
            {
                DNDS_FIELD(gamma,       "Ratio of specific heats",
                           DNDS::Config::range(1.0));
                DNDS_FIELD(Rgas,        "Specific gas constant",
                           DNDS::Config::range(0.0));
                DNDS_FIELD(muGas,       "Dynamic viscosity",
                           DNDS::Config::range(0.0));
                DNDS_FIELD(prGas,       "Prandtl number",
                           DNDS::Config::range(0.0));
                DNDS_FIELD(TRef,        "Reference temperature (K)");
                DNDS_FIELD(CSutherland, "Sutherland constant (K)");
                DNDS_FIELD(muModel,     "Viscosity model: 0=constant, 1=sutherland, 2=constant_nu");
                // CpGas is derived: recomputed after deserialization
                config.post_read([](T &s) { s.recomputeDerived(); });
            }

            /// @brief Recompute derived quantities (CpGas) from gamma and Rgas after deserialization.
            void recomputeDerived()
            {
                CpGas = Rgas * gamma / (gamma - 1);
            }
        } idealGasProperty; ///< Ideal gas thermodynamic property configuration.

        /***************************************************************************************************/
        // end of setting entries
        /***************************************************************************************************/

        int _nVars = 0; ///< Runtime nVars, not serialized. Set by ctor, preserved across from_json.
        Eigen::Vector<real, -1> refU;     ///< Reference conservative state (derived from farFieldStaticValue).
        Eigen::Vector<real, -1> refUPrim; ///< Reference primitive state (derived from farFieldStaticValue).

        DNDS_DECLARE_CONFIG(EulerEvaluatorSettings)
        {
            DNDS_FIELD(useScalarJacobian,       "Use scalar Jacobian approximation");
            DNDS_FIELD(useRoeJacobian,          "Use Roe-based Jacobian");
            DNDS_FIELD(noRsOnWall,              "Disable Riemann solver on wall boundaries");
            DNDS_FIELD(noGRPOnWall,             "Disable GRP on wall boundaries");
            DNDS_FIELD(ignoreSourceTerm,        "Ignore source terms");
            DNDS_FIELD(direct2ndRecMethod,      "Direct 2nd-order reconstruction method");
            DNDS_FIELD(specialBuiltinInitializer, "Special built-in initializer code");
            DNDS_FIELD(uRecAlphaCompressPower,  "uRec alpha compression power");
            DNDS_FIELD(uRecBetaCompressPower,   "uRec beta compression power");
            DNDS_FIELD(forceVolURecBeta,        "Force volume uRec beta");
            DNDS_FIELD(ppEpsIsRelaxed,          "Positivity-preserving epsilon is relaxed");
            DNDS_FIELD(RANSBottomLimit,         "RANS variable bottom limit",
                       DNDS::Config::range(0.0));
            config.field_alias(&T::rsType,      "riemannSolverType",
                               "Riemann solver type");
            config.field_alias(&T::rsTypeAux,   "riemannSolverTypeAux",
                               "Auxiliary Riemann solver type");
            config.field_alias(&T::rsTypeWall,  "riemannSolverTypeWall",
                               "Wall Riemann solver type");
            DNDS_FIELD(rsFixScale,              "Riemann solver entropy fix scale");
            DNDS_FIELD(rsIncFScale,             "Riemann solver increment flux scale");
            DNDS_FIELD(rsMeanValueEig,          "Riemann solver mean-value eigenvalue mode");
            DNDS_FIELD(rsRotateScheme,          "Riemann solver rotation scheme");
            DNDS_FIELD(minWallDist,             "Minimum wall distance clamp",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(wallDistExection,        "Wall distance execution mode: 0=parallel, 1=serial");
            DNDS_FIELD(wallDistRefineMax,       "Wall distance max refinement");
            DNDS_FIELD(wallDistScheme,          "Wall distance computation scheme");
            DNDS_FIELD(wallDistCellLoadSize,    "Wall distance cell load batch size",
                       DNDS::Config::range(1));
            DNDS_FIELD(wallDistIter,            "Wall distance solver iterations",
                       DNDS::Config::range(1));
            DNDS_FIELD(wallDistLinSolver,       "Wall distance linear solver: 0=jacobi, 1=gmres");
            DNDS_FIELD(wallDistResTol,          "Wall distance residual tolerance",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(wallDistIterStart,       "Wall distance solver start iteration",
                       DNDS::Config::range(0));
            DNDS_FIELD(wallDistPoissonP,        "Wall distance Poisson equation power");
            DNDS_FIELD(wallDistDTauScale,       "Wall distance pseudo-time scale",
                       DNDS::Config::range(0.0));
            DNDS_FIELD(wallDistNJacobiSweep,    "Wall distance Jacobi sweep count",
                       DNDS::Config::range(1));
            DNDS_FIELD(SADESScale,              "SA-DES length scale");
            DNDS_FIELD(SADESMode,               "SA-DES mode");
            DNDS_FIELD(SAVersion,               "SA variant: 0=current (cRot=2, corrected |S|, min(P,0) Jacobian), 1=legacy/pre-31578ce (cRot=1, Frobenius SS, P*0 Jacobian)");
            DNDS_FIELD(ransModel,               "RANS turbulence model");
            DNDS_FIELD(ransUseQCR,              "Use QCR correction for RANS");
            DNDS_FIELD(ransSARotCorrection,     "SA rotation correction");
            DNDS_FIELD(ransEigScheme,           "RANS eigenvalue scheme");
            DNDS_FIELD(ransForce2nd,            "Force 2nd-order RANS");
            DNDS_FIELD(ransSource2nd,           "RANS source 2nd-order");
            DNDS_FIELD(source2nd,               "Source term 2nd-order");
            DNDS_FIELD(usePrimGradInVisFlux,    "Use primitive gradient in viscous flux");
            DNDS_FIELD(useSourceGradFixGG,      "Use source gradient fix for Green-Gauss");
            DNDS_FIELD(nCentralSmoothStep,      "Central smoothing steps",
                       DNDS::Config::range(0));
            DNDS_FIELD(centralSmoothEps,        "Central smoothing epsilon");
            DNDS_FIELD(constMassForce,          "Constant mass force vector (3D)");
            config.field_section(&T::frameConstRotation, "frameConstRotation",
                                 "Constant-rotation reference frame settings");
            config.field_section(&T::cLDriverSettings,   "cLDriverSettings",
                                 "CL driver settings");
            DNDS_FIELD(cLDriverBCNames,         "Boundary names for CL driver force integration");
            DNDS_FIELD(farFieldStaticValue,     "Far-field static value vector (size = nVars)");
            config.template field_array_of<BoxInitializer>(
                &T::boxInitializers, "boxInitializers",
                "Box region initializers");
            config.template field_array_of<PlaneInitializer>(
                &T::planeInitializers, "planeInitializers",
                "Plane region initializers");
            config.template field_array_of<ExprtkInitializer>(
                &T::exprtkInitializers, "exprtkInitializers",
                "Exprtk expression initializers");
            config.field_section(&T::idealGasProperty, "idealGasProperty",
                                 "Ideal gas thermodynamic properties");

            // Cross-field checks
            config.check("useScalarJacobian and useRoeJacobian are mutually exclusive",
                         [](const T &s) { return !(s.useScalarJacobian && s.useRoeJacobian); });
            config.check("ransModel must not be RANS_Unknown",
                         [](const T &s) { return s.ransModel != RANS_Unknown; });

            // Post-read hook: finalize derived quantities using stored _nVars
            config.post_read([](T &s) { s.finalize(); });
        }

        /// @brief Default constructor (used for schema emission; _nVars remains 0).
        EulerEvaluatorSettings() = default;

        /**
         * @brief Construct with a known variable count and set model-appropriate defaults.
         *
         * If the model includes SA or 2-equation RANS traits, the default ransModel
         * is set accordingly. The farFieldStaticValue is sized to @p nVars and
         * initialized to a default freestream state.
         *
         * @param nVars  Number of conservative variables for this model.
         */
        EulerEvaluatorSettings(int nVars) : _nVars(nVars)
        {
            if constexpr (Traits::hasSA)
            {
                ransModel = RANSModel::RANS_SA;
            }
            if constexpr (Traits::has2EQ)
            {
                ransModel = RANSModel::RANS_KOWilcox;
            }
            farFieldStaticValue.setOnes(nVars);
            DNDS_assert(nVars > I4);
            farFieldStaticValue(0) = 1;
            farFieldStaticValue(Eigen::seq(Eigen::fix<0>, Eigen::fix<I4>)).setZero();
            farFieldStaticValue(I4) = 2.5;
        }

        /**
         * @brief Post-deserialization finalization: cross-field validation and derived quantities.
         *
         * Checks dimensional consistency of farFieldStaticValue, boxInitializers, and
         * planeInitializers against _nVars. Normalizes the rotation axis if the rotating
         * frame is enabled. Computes refU and refUPrim from the far-field state and ideal
         * gas properties for use as reference scales in the solver.
         *
         * Uses the stored _nVars set by the constructor. Called automatically by the
         * post_read hook after from_json, or explicitly after copy-construction.
         * If _nVars <= 0 (e.g. default-constructed for schema emission), this is a no-op.
         */
        void finalize()
        {
            int nVars = _nVars;
            if (nVars <= 0)
                return; // skip finalize if nVars not set (e.g. schema emission default-ctor)
            DNDS_assert(constMassForce.size() == 3);
            DNDS_assert(farFieldStaticValue.size() == nVars);
            if (constMassForce.norm() || frameConstRotation.enabled ||
                std::unordered_set<EulerModel>{NS_SA, NS_SA_3D, NS_2EQ, NS_2EQ_3D}.count(model))
                DNDS_assert_info(!ignoreSourceTerm,
                                "you have set source term, do not use ignoreSourceTerm! ");
            if (frameConstRotation.enabled)
                frameConstRotation.axis.normalize();
            for (auto &box : boxInitializers)
                DNDS_assert_info(box.v.size() == nVars, "box initial value dimension incorrect");
            for (auto &plane : planeInitializers)
                DNDS_assert_info(plane.v.size() == nVars, "plane initial value dimension incorrect");

            // Compute reference values
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            refU = farFieldStaticValue;
            refUPrim = refU;
            Gas::IdealGasThermalConservative2Primitive<dim>(refU, refUPrim, idealGasProperty.gamma);
            DNDS_assert(refUPrim(I4) > 0 && refUPrim(0) > 0);
            real a = std::sqrt(idealGasProperty.gamma * refUPrim(I4) / (refUPrim(0) + verySmallReal));
            refU(Seq123).setConstant(refU(Seq123).norm() + a);
            refUPrim(Seq123).setConstant(refUPrim(Seq123).norm());
        }
    };
}