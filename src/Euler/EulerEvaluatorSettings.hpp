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
#include "DNDS/Serializer/JsonUtil.hpp"
#include "DNDS/Config/ConfigParam.hpp"

#include "Euler.hpp"
#include "Gas.hpp"
#include "CLDriver.hpp"
#include <cmath>
#include <unordered_set>
#include <string>
#include <limits>

namespace DNDS::Euler
{
    enum class StateValueOrigin
    {
        None,
        Cons,
        ConsSensible,
        PrimRhoP,
        PrimRhoT,
        PrimTP,
        ConsPhy,
        ConsSensiblePhy,
        PrimRhoPPhy,
        PrimRhoTPhy,
        PrimTPPhy,
        NonState,
        Invalid,
    };

    inline std::string StateValueOriginName(StateValueOrigin origin)
    {
        switch (origin)
        {
        case StateValueOrigin::Cons:
            return "cons";
        case StateValueOrigin::ConsSensible:
            return "consSensible";
        case StateValueOrigin::PrimRhoP:
            return "primRhoP";
        case StateValueOrigin::PrimRhoT:
            return "primRhoT";
        case StateValueOrigin::PrimTP:
            return "primTP";
        case StateValueOrigin::ConsPhy:
            return "cons_phy";
        case StateValueOrigin::ConsSensiblePhy:
            return "consSensible_phy";
        case StateValueOrigin::PrimRhoPPhy:
            return "primRhoP_phy";
        case StateValueOrigin::PrimRhoTPhy:
            return "primRhoT_phy";
        case StateValueOrigin::PrimTPPhy:
            return "primTP_phy";
        case StateValueOrigin::NonState:
            return "nonState";
        case StateValueOrigin::Invalid:
            return "invalid";
        default:
            return "none";
        }
    }

    inline StateValueOrigin StateValueOriginFromName(const std::string &name)
    {
        if (name == "cons")
            return StateValueOrigin::Cons;
        if (name == "consSensible")
            return StateValueOrigin::ConsSensible;
        if (name == "primRhoP")
            return StateValueOrigin::PrimRhoP;
        if (name == "primRhoT")
            return StateValueOrigin::PrimRhoT;
        if (name == "primTP")
            return StateValueOrigin::PrimTP;
        if (name == "cons_phy")
            return StateValueOrigin::ConsPhy;
        if (name == "consSensible_phy")
            return StateValueOrigin::ConsSensiblePhy;
        if (name == "primRhoP_phy")
            return StateValueOrigin::PrimRhoPPhy;
        if (name == "primRhoT_phy")
            return StateValueOrigin::PrimRhoTPhy;
        if (name == "primTP_phy")
            return StateValueOrigin::PrimTPPhy;
        if (name == "nonState")
            return StateValueOrigin::NonState;
        return StateValueOrigin::None;
    }

    inline nlohmann::ordered_json StateValueSchema(const std::string &desc, bool allowNonState = false)
    {
        using json = nlohmann::ordered_json;
        json stateTypes = json::array({"cons", "consSensible", "primRhoP", "primRhoT", "primTP",
                                       "cons_phy", "consSensible_phy", "primRhoP_phy", "primRhoT_phy", "primTP_phy"});
        if (allowNonState)
            stateTypes.push_back("nonState");

        json numArray{{"type", "array"}, {"items", json{{"type", "number"}}}};
        json tagged;
        tagged["type"] = "object";
        tagged["required"] = json::array({"type", "state"});
        tagged["additionalProperties"] = false;
        tagged["properties"] = json::object();
        tagged["properties"]["type"] = json{{"type", "string"}, {"enum", stateTypes}};
        tagged["properties"]["state"] = numArray;

        json legacy = numArray;
        legacy["description"] = "Legacy plain array; interpreted as consSensible.";

        json s;
        s["description"] = desc + std::string(" Canonical object form is {type, state}. Merge-patch warning: patch type and state together; changing only one reinterprets stale data.");
        s["default"] = json{{"type", "consSensible"}, {"state", json::array()}};
        s["oneOf"] = json::array({legacy, tagged});
        return s;
    }

    inline nlohmann::ordered_json StateValueTypeNameSchema(const std::string &desc)
    {
        using json = nlohmann::ordered_json;
        return json{{"type", "string"},
                    {"description", desc},
                    {"enum", json::array({"cons", "consSensible", "primRhoP", "primRhoT", "primTP",
                                          "cons_phy", "consSensible_phy", "primRhoP_phy", "primRhoT_phy", "primTP_phy"})}};
    }

    struct StateValue
    {
        Eigen::Vector<real, -1> cons;
        Eigen::Vector<real, -1> consSensible;
        Eigen::Vector<real, -1> primRhoP;
        Eigen::Vector<real, -1> primRhoT;
        Eigen::Vector<real, -1> primTP;
        Eigen::Vector<real, -1> cons_phy;
        Eigen::Vector<real, -1> consSensible_phy;
        Eigen::Vector<real, -1> primRhoP_phy;
        Eigen::Vector<real, -1> primRhoT_phy;
        Eigen::Vector<real, -1> primTP_phy;
        Eigen::Vector<real, -1> nonState;
        StateValueOrigin originType = StateValueOrigin::None;

        static bool filled(const Eigen::Vector<real, -1> &v) { return v.size() > 0 && v.allFinite(); }

        const Eigen::Vector<real, -1> &originVector() const
        {
            switch (originType)
            {
            case StateValueOrigin::Cons:
                return cons;
            case StateValueOrigin::ConsSensible:
                return consSensible;
            case StateValueOrigin::PrimRhoP:
                return primRhoP;
            case StateValueOrigin::PrimRhoT:
                return primRhoT;
            case StateValueOrigin::PrimTP:
                return primTP;
            case StateValueOrigin::ConsPhy:
                return cons_phy;
            case StateValueOrigin::ConsSensiblePhy:
                return consSensible_phy;
            case StateValueOrigin::PrimRhoPPhy:
                return primRhoP_phy;
            case StateValueOrigin::PrimRhoTPhy:
                return primRhoT_phy;
            case StateValueOrigin::PrimTPPhy:
                return primTP_phy;
            case StateValueOrigin::NonState:
                return nonState;
            default:
                return cons;
            }
        }

        Eigen::Vector<real, -1> &originVectorMutable(StateValueOrigin origin)
        {
            switch (origin)
            {
            case StateValueOrigin::Cons:
                return cons;
            case StateValueOrigin::ConsSensible:
                return consSensible;
            case StateValueOrigin::PrimRhoP:
                return primRhoP;
            case StateValueOrigin::PrimRhoT:
                return primRhoT;
            case StateValueOrigin::PrimTP:
                return primTP;
            case StateValueOrigin::ConsPhy:
                return cons_phy;
            case StateValueOrigin::ConsSensiblePhy:
                return consSensible_phy;
            case StateValueOrigin::PrimRhoPPhy:
                return primRhoP_phy;
            case StateValueOrigin::PrimRhoTPhy:
                return primRhoT_phy;
            case StateValueOrigin::PrimTPPhy:
                return primTP_phy;
            case StateValueOrigin::NonState:
                return nonState;
            default:
                return cons;
            }
        }

        void keepOnlyOrigin()
        {
            auto clearUnless = [&](Eigen::Vector<real, -1> &v, StateValueOrigin origin)
            {
                if (originType != origin)
                    v.resize(0);
            };
            clearUnless(cons, StateValueOrigin::Cons);
            clearUnless(consSensible, StateValueOrigin::ConsSensible);
            clearUnless(primRhoP, StateValueOrigin::PrimRhoP);
            clearUnless(primRhoT, StateValueOrigin::PrimRhoT);
            clearUnless(primTP, StateValueOrigin::PrimTP);
            clearUnless(cons_phy, StateValueOrigin::ConsPhy);
            clearUnless(consSensible_phy, StateValueOrigin::ConsSensiblePhy);
            clearUnless(primRhoP_phy, StateValueOrigin::PrimRhoPPhy);
            clearUnless(primRhoT_phy, StateValueOrigin::PrimRhoTPhy);
            clearUnless(primTP_phy, StateValueOrigin::PrimTPPhy);
            clearUnless(nonState, StateValueOrigin::NonState);
        }

        void fillMissingWithNaN(int nVars)
        {
            auto fillOne = [&](Eigen::Vector<real, -1> &v)
            {
                if (v.size() == 0)
                    v.setConstant(nVars, std::numeric_limits<real>::quiet_NaN());
            };
            fillOne(cons);
            fillOne(consSensible);
            fillOne(primRhoP);
            fillOne(primRhoT);
            fillOne(primTP);
            fillOne(cons_phy);
            fillOne(consSensible_phy);
            fillOne(primRhoP_phy);
            fillOne(primRhoT_phy);
            fillOne(primTP_phy);
            fillOne(nonState);
        }

        void checkSize(int nVars, const std::string &name) const
        {
            auto checkOne = [&](const Eigen::Vector<real, -1> &v, const char *key)
            {
                DNDS_check_throw_info(v.size() == 0 || v.size() == nVars,
                                      fmt::format("{} {} dim {} != {}", name, key, v.size(), nVars));
            };
            checkOne(cons, "cons");
            checkOne(consSensible, "consSensible");
            checkOne(primRhoP, "primRhoP");
            checkOne(primRhoT, "primRhoT");
            checkOne(primTP, "primTP");
            checkOne(cons_phy, "cons_phy");
            checkOne(consSensible_phy, "consSensible_phy");
            checkOne(primRhoP_phy, "primRhoP_phy");
            checkOne(primRhoT_phy, "primRhoT_phy");
            checkOne(primTP_phy, "primTP_phy");
            checkOne(nonState, "nonState");
        }

        friend void from_json(const nlohmann::ordered_json &j, StateValue &v)
        {
            v = StateValue{};
            if (j.is_array())
            {
                v.consSensible = j.get<Eigen::VectorXd>();
                v.originType = StateValueOrigin::ConsSensible;
                return;
            }
            DNDS_check_throw_info(j.is_object(), "StateValue must be an array or object");
            for (auto it = j.begin(); it != j.end(); ++it)
                DNDS_check_throw_info(it.key() == "type" || it.key() == "state",
                                      fmt::format("StateValue object only accepts keys 'type' and 'state', got [{}]", it.key()));
            DNDS_check_throw_info(j.contains("type") && j.contains("state"),
                                  "StateValue object must use canonical form {\"type\": string, \"state\": [...]}"
                                  "; old keyed forms like {\"primTP_phy\": [...]} are not accepted");
            DNDS_check_throw_info(j.at("type").is_string(), "StateValue.type must be a string");
            DNDS_check_throw_info(j.at("state").is_array(), "StateValue.state must be an array");
            v.originType = StateValueOriginFromName(j.at("type").get<std::string>());
            DNDS_check_throw_info(v.originType != StateValueOrigin::None && v.originType != StateValueOrigin::Invalid,
                                  fmt::format("unknown StateValue.type [{}]", j.at("type").get<std::string>()));
            v.originVectorMutable(v.originType) = j.at("state").get<Eigen::VectorXd>();
            v.keepOnlyOrigin();
        }

        friend void to_json(nlohmann::ordered_json &j, const StateValue &v)
        {
            j = nlohmann::ordered_json::object();
            j["type"] = StateValueOriginName(v.originType);
            if (v.originType != StateValueOrigin::None && v.originType != StateValueOrigin::Invalid)
                j["state"] = v.originVector();
            else
                j["state"] = nlohmann::ordered_json::array();
        }
    };

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
        using Traits = EulerModelTraits<model>;             ///< Compile-time model traits.
        static const int nVarsFixed = getnVarsFixed(model); ///< Compile-time variable count.
        static const int dim = getDim_Fixed(model);         ///< Physical dimension (2 or 3).
        static const int gDim = getGeomDim_Fixed(model);    ///< Geometric dimension (may differ for axi-symmetric).
        static const auto I4 = dim + 1;                     ///< Index of the energy equation in the state vector.

        /// @name Jacobian Options
        /// @{
        bool useScalarJacobian = false; ///< Use scalar (diagonal) Jacobian approximation instead of block.
        bool useRoeJacobian = false;    ///< Use Roe-linearization-based Jacobian.
        bool noRsOnWall = false;        ///< Disable the Riemann solver on wall boundary faces.
        bool noGRPOnWall = false;       ///< Disable the Generalized Riemann Problem (GRP) on wall faces.
        bool ignoreSourceTerm = false;  ///< Completely ignore source terms (must be false when RANS or body forces are active).
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

        real RANSTopLimit = 1e5;     ///< Upper clamp for SA nutilde
        real RANSBottomLimit = 0.01; ///< Lower clamp for RANS turbulence variables.

        /// @name Riemann Solver Configuration
        /// @{
        Gas::RiemannSolverType rsType = Gas::Roe;           ///< Primary Riemann solver type.
        Gas::RiemannSolverType rsTypeAux = Gas::UnknownRS;  ///< Auxiliary Riemann solver type (UnknownRS = same as primary).
        Gas::RiemannSolverType rsTypeWall = Gas::UnknownRS; ///< Wall-face Riemann solver type (UnknownRS = same as primary).
        real rsFixScale = 1;                                ///< Entropy-fix scaling factor for the Riemann solver.
        real rsIncFScale = 1;                               ///< Incremental flux scaling factor.
        int rsMeanValueEig = 0;                             ///< Mean-value eigenvalue computation mode.
        int rsRotateScheme = 0;                             ///< Riemann solver rotation scheme selector.
        /// @}

        /// @name Wall-Distance Computation
        /// @{
        real minWallDist = 1e-12;             ///< Minimum wall distance clamp (avoids singularities).
        int wallDistExection = 0;             ///< Execution mode: 0 = parallel, 1 = serial.
        real wallDistRefineMax = 1;           ///< Maximum wall-distance refinement factor.
        int wallDistScheme = 0;               ///< Wall-distance computation scheme selector.
        int wallDistCellLoadSize = 1024 * 32; ///< Cell batch size for wall-distance computation.
        int wallDistIter = 1000;              ///< Maximum iterations for the wall-distance solver.
        int wallDistLinSolver = 0;            ///< Linear solver: 0 = Jacobi, 1 = GMRES.
        real wallDistResTol = 1e-4;           ///< Residual tolerance for wall-distance convergence.
        int wallDistIterStart = 100;          ///< Starting iteration count for the wall-distance solver.
        int wallDistPoissonP = 2;             ///< Poisson equation power in the wall-distance PDE.
        real wallDistDTauScale = 100.;        ///< Pseudo-time step scaling for wall-distance solver.
        int wallDistNJacobiSweep = 10;        ///< Number of Jacobi sweeps per wall-distance iteration.
        /// @}

        /// @name RANS / DES Configuration
        /// @{
        real SADESScale = veryLargeReal; ///< SA-DES length scale (veryLargeReal effectively disables DES).
        int SADESMode = 1;               ///< SA-DES mode selector (1 = DDES, etc.).
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
        int ransUseQCR = 0;                         ///< Enable QCR (Quadratic Constitutive Relation) correction.
        int ransSARotCorrection = 1;                ///< SA rotation/curvature correction mode.
        int ransEigScheme = 0;                      ///< Eigenvalue computation scheme for RANS.
        int ransForce2nd = 0;                       ///< Force 2nd-order accuracy for RANS variables.
        int ransSource2nd = 0;                      ///< Enable 2nd-order RANS source term discretization.
        /// @}

        /// @name Viscous Flux and Source Options
        /// @{
        int source2nd = 0;                    ///< Enable 2nd-order source term discretization.
        int usePrimGradInVisFlux = 0;         ///< Use primitive-variable gradients in viscous flux.
        int useSourceGradFixGG = 0;           ///< Apply Green-Gauss gradient fix for source terms.
        int nCentralSmoothStep = 0;           ///< Number of central-difference smoothing steps.
        real centralSmoothEps = 0.5;          ///< Epsilon for central smoothing.
        int pointImplicitSourceUpdateOut = 0; ///< Print point-implicit source-update Newton residual ratios.
        struct ReactorStepSettings
        {
            real rtol = 1e-10;       ///< Cantera reactor relative tolerance.
            real atol = 1e-18;       ///< Cantera reactor absolute tolerance.
            int maxOrder = 0;        ///< CVODE max order; <=0 leaves Cantera default.
            int maxSteps = 10000000; ///< CVODE max internal steps.

            DNDS_DECLARE_CONFIG(ReactorStepSettings)
            {
                // clang-format off
                DNDS_FIELD(rtol,     "Cantera reactor relative tolerance", DNDS::Config::range(0.0));
                DNDS_FIELD(atol,     "Cantera reactor absolute tolerance", DNDS::Config::range(0.0));
                DNDS_FIELD(maxOrder, "Cantera reactor max order; <=0 leaves default");
                DNDS_FIELD(maxSteps, "Cantera reactor max internal steps", DNDS::Config::range(1));
                // clang-format on
            }
        } reactorStepSettings;                                                   ///< Settings for direct Cantera source substeps.
        real reactiveSourceScale = 1.0;                                          ///< Multiplier for reactive source RHS and Jacobian.
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
            bool enabled = false;                        ///< Enable the rotating frame.
            Geom::tPoint axis = Geom::tPoint{0, 0, 1};   ///< Rotation axis (unit vector; normalized in finalize()).
            Geom::tPoint center = Geom::tPoint{0, 0, 0}; ///< Center of rotation [x, y, z].
            real rpm = 0;                                ///< Rotational speed in revolutions per minute.

            /// @brief Compute angular velocity magnitude (rad/s) from RPM.
            /// @return Omega = rpm * 2π / 60.
            real Omega() const
            {
                return rpm * (2 * pi / 60.);
            }

            /// @brief Return the angular velocity vector (axis * Omega).
            /// @return 3-D omega vector aligned with the rotation axis.
            Geom::tPoint vOmega() const
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
                // clang-format off
                DNDS_FIELD(enabled, "Enable constant-rotation reference frame");
                DNDS_FIELD(axis,    "Rotation axis (unit vector)");
                DNDS_FIELD(center,  "Rotation center coordinates");
                DNDS_FIELD(rpm,     "Rotational speed in RPM");
                // clang-format on
            }
        } frameConstRotation;                     ///< Rotating reference frame configuration.
        CLDriverSettings cLDriverSettings;        ///< Lift-coefficient (CL) driver settings.
        std::vector<std::string> cLDriverBCNames; ///< Boundary zone names for CL driver force integration.
        StateValue farFieldStaticValue;           ///< Far-field reference state; resolved to conservative total before use.
        /**
         * @brief Axis-aligned box region for initial condition specification.
         *
         * Cells whose centroids lie within [x0,x1]×[y0,y1]×[z0,z1] are
         * initialized to the state vector @c v.
         */
        struct BoxInitializer
        {
            real x0{0}, x1{0}, y0{0}, y1{0}, z0{0}, z1{0}; ///< Box bounds [min, max] per axis.
            StateValue v;                                  ///< Initial state; resolved to conservative total before use.

            DNDS_DECLARE_CONFIG(BoxInitializer)
            {
                // clang-format off
                DNDS_FIELD(x0, "Box x-min");
                DNDS_FIELD(x1, "Box x-max");
                DNDS_FIELD(y0, "Box y-min");
                DNDS_FIELD(y1, "Box y-max");
                DNDS_FIELD(z0, "Box z-min");
                DNDS_FIELD(z1, "Box z-max");
                config.field_schema(&T::v, "v", "Initial state value (size = nVars)",
                                    []() { return StateValueSchema("Initial state value (size = nVars)"); });
                // clang-format on
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
            StateValue v;                ///< Initial state; resolved to conservative total before use.

            DNDS_DECLARE_CONFIG(PlaneInitializer)
            {
                // clang-format off
                DNDS_FIELD(a, "Plane normal x-component");
                DNDS_FIELD(b, "Plane normal y-component");
                DNDS_FIELD(c, "Plane normal z-component");
                DNDS_FIELD(h, "Plane offset");
                config.field_schema(&T::v, "v", "Initial state value (size = nVars)",
                                    []() { return StateValueSchema("Initial state value (size = nVars)"); });
                // clang-format on
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
            std::vector<std::string> exprs;     ///< ExprTk expression lines (concatenated with newlines).
            std::string stateType = "primRhoP"; ///< ExprTk vector state convention (StateValue type name, excluding nonState).

            DNDS_DECLARE_CONFIG(ExprtkInitializer)
            {
                // clang-format off
                DNDS_FIELD(exprs, "Expression lines (concatenated with newlines)");
                config.field_schema(&T::stateType, "stateType",
                                    "ExprTk state convention: cons, consSensible, primRhoP, primRhoT, primTP, or *_phy variants.",
                                    []() { return StateValueTypeNameSchema("ExprTk state convention: cons, consSensible, primRhoP, primRhoT, primTP, or *_phy variants."); });
                // clang-format on
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
         */
        struct IdealGasProperty
        {
            real gamma = 1.4;
            real Rgas = 287; ///< physical gas constant R_phys [J/(kg·K)]; consumed via toCode() → R_code = R_phys·T0/U0²
            real muGas = 1;  ///< dynamic viscosity [Pa·s] physical (μ_phys), code-scaled via μ_0 = ρ0·U0·L0
            real prGas = 0.72;
            real TRef = 273.15;
            real CSutherland = 110.4;
            int muModel = 1;

            /// Reference scales for dimensional-physical conversion.
            /// R_code = R_phys / R0  where R0 = U0² / T0.
            /// p0 = rho0 · U0².
            real T0 = 1;   ///< Reference temperature (K).
            real rho0 = 1; ///< Reference density (kg/m³).
            real U0 = 1;   ///< Reference velocity (m/s).
            real L0 = 1;   ///< Reference length (m).

            DNDS_DECLARE_CONFIG(IdealGasProperty)
            {
                // clang-format off
                DNDS_FIELD(gamma,       "Ratio of specific heats. Must be finite and > 1.",
                           DNDS::Config::range(1.0 + std::numeric_limits<real>::epsilon()));
                DNDS_FIELD(Rgas,        "Specific gas constant. Must be finite and > 0.",
                           DNDS::Config::range(std::numeric_limits<real>::min()));
                DNDS_FIELD(muGas,       "Dynamic viscosity (Pa·s)",
                           DNDS::Config::range(0.0));
                DNDS_FIELD(prGas,       "Prandtl number",
                           DNDS::Config::range(0.0));
                DNDS_FIELD(TRef,        "Reference temperature (K)");
                DNDS_FIELD(CSutherland, "Sutherland constant (K)");
                DNDS_FIELD(muModel,     "Viscosity model: 0=constant, 1=sutherland, 2=constant_nu");
                DNDS_FIELD(T0,          "Reference temperature (K). Must be finite and > 0.", DNDS::Config::range(std::numeric_limits<real>::min()));
                DNDS_FIELD(rho0,        "Reference density (kg/m^3). Must be finite and > 0.", DNDS::Config::range(std::numeric_limits<real>::min()));
                DNDS_FIELD(U0,          "Reference velocity (m/s). Must be finite and > 0.", DNDS::Config::range(std::numeric_limits<real>::min()));
                DNDS_FIELD(L0,          "Reference length (m). Must be finite and > 0.", DNDS::Config::range(std::numeric_limits<real>::min()));
                config.post_read([](T &s) { s.validateScales(); });
                // clang-format on
            }

            /// @brief Constant-pressure heat capacity from gamma and Rgas.
            real CpGas() const
            {
                return Rgas * gamma / (gamma - 1);
            }

            void validateScales() const
            {
                DNDS_check_throw_info(std::isfinite(T0) && T0 > 0, "idealGasProperty.T0 must be finite and > 0");
                DNDS_check_throw_info(std::isfinite(rho0) && rho0 > 0, "idealGasProperty.rho0 must be finite and > 0");
                DNDS_check_throw_info(std::isfinite(U0) && U0 > 0, "idealGasProperty.U0 must be finite and > 0");
                DNDS_check_throw_info(std::isfinite(L0) && L0 > 0, "idealGasProperty.L0 must be finite and > 0");
                DNDS_check_throw_info(std::isfinite(gamma) && gamma > 1, "idealGasProperty.gamma must be finite and > 1");
                DNDS_check_throw_info(std::isfinite(Rgas) && Rgas > 0, "idealGasProperty.Rgas must be finite and > 0");
            }
        } idealGasProperty; ///< Ideal gas thermodynamic property configuration.

        /**
         * @brief Reactive flow configuration.
         *
         * When @c enabled is true, the solver activates multi-species reactive
         * flow with chemical source terms and multi-species thermodynamics.
         * Phase 1: structural placeholder — chemistry evaluation wired in Phase 2.
         */
        struct ReactiveFlowSettings
        {
            bool enabled = false;
            std::string mechanismFile;
            std::string thermoFile;
            std::string transportModel = "MixtureAveraged";
            real CFLScale = 1.0;
            real chemRelaxEps = 1e-3;
            real chemAbsTol = 1e-10;
            real TBase = 0.0;
            int nSpeciesOverride = 0;

            DNDS_DECLARE_CONFIG(ReactiveFlowSettings)
            {
                DNDS_FIELD(enabled, "Enable reactive flow (multi-species + chemistry)");
                DNDS_FIELD(mechanismFile, "CHEMKIN-format mechanism YAML path");
                DNDS_FIELD(thermoFile, "Reserved; currently unused. Mechanism YAML supplies thermodynamics.");
                DNDS_FIELD(transportModel, "Cantera transport model requested by reactive flow; currently only mixture-averaged is implemented.");
                DNDS_FIELD(CFLScale, "CFL reduction factor for stiff chemistry",
                           DNDS::Config::range(0.0));
                DNDS_FIELD(chemRelaxEps, "Pseudo-transient relaxation epsilon",
                           DNDS::Config::range(0.0));
                DNDS_FIELD(chemAbsTol, "Absolute species tolerance",
                           DNDS::Config::range(0.0));
                DNDS_FIELD(TBase, "Base temperature [K] for reactive sensible-energy bookkeeping; <=0 uses the minimum per-species Cantera bound");
                DNDS_FIELD(nSpeciesOverride, "Reserved; currently unused. Species count is read from mechanism.");
            }
        } reactiveFlow; ///< Reactive flow settings.

        /***************************************************************************************************/
        // end of setting entries
        /***************************************************************************************************/

        int _nVars = 0;                   ///< Runtime nVars, not serialized. Set by ctor, preserved across from_json.
        Eigen::Vector<real, -1> refU;     ///< Reference conservative state (derived from farFieldStaticValue).
        Eigen::Vector<real, -1> refUPrim; ///< Reference primitive state (derived from farFieldStaticValue).

        DNDS_DECLARE_CONFIG(EulerEvaluatorSettings)
        {
            // clang-format off
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
            DNDS_FIELD(RANSTopLimit,         "RANS variable top limit, currently for SA nutilde",
                       DNDS::Config::range(0.0));
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
            DNDS_FIELD(pointImplicitSourceUpdateOut, "Print point-implicit source-update Newton residual ratio min/max: 0=off, 1=on");
            config.field_section(&T::reactorStepSettings, "reactorStepSettings",
                                 "Cantera reactor settings for direct source substeps");
            DNDS_FIELD(reactiveSourceScale,     "Scale reactive source RHS and Jacobian directly; use 0 for non-reactive debugging");
            DNDS_FIELD(constMassForce,          "Constant mass force vector (3D)");
            config.field_section(&T::frameConstRotation, "frameConstRotation",
                                 "Constant-rotation reference frame settings");
            config.field_section(&T::cLDriverSettings,   "cLDriverSettings",
                                 "CL driver settings");
            DNDS_FIELD(cLDriverBCNames,         "Boundary names for CL driver force integration");
            config.field_schema(&T::farFieldStaticValue, "farFieldStaticValue",
                                "Far-field state value (size = nVars)",
                                []() { return StateValueSchema("Far-field state value (size = nVars)"); });
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
            config.field_section(&T::reactiveFlow,     "reactiveFlow",
                                  "Reactive flow settings (multi-species chemistry)");

            // Cross-field checks
            config.check("useScalarJacobian and useRoeJacobian are mutually exclusive",
                         [](const T &s) { return !(s.useScalarJacobian && s.useRoeJacobian); });
            config.check("ransModel must not be RANS_Unknown",
                         [](const T &s) { return s.ransModel != RANS_Unknown; });

            // Post-read hook: finalize derived quantities using stored _nVars
            config.post_read([](T &s) { s.finalize(); });
            // clang-format on
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
            DNDS_assert(nVars > I4);
            farFieldStaticValue.consSensible.setOnes(nVars);
            farFieldStaticValue.consSensible(0) = 1;
            farFieldStaticValue.consSensible(Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>)).setZero();
            farFieldStaticValue.consSensible(I4) = 2.5;
            farFieldStaticValue.consSensible(Eigen::seq(I4 + 1, nVars - 1)).setZero();
            farFieldStaticValue.originType = StateValueOrigin::ConsSensible;
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
            DNDS_check_throw_info(!reactiveFlow.enabled || model == NS_EX || model == NS_EX_3D,
                                  "reactiveFlow.enabled is only supported for eulerEX/eulerEX3D models");
            DNDS_check_throw_info(!reactiveFlow.enabled || specialBuiltinInitializer == 0,
                                  "reactiveFlow.enabled does not support specialBuiltinInitializer; use explicit StateValue/ExprTk initialization");
            DNDS_check_throw_info(!reactiveFlow.enabled || !reactiveFlow.mechanismFile.empty(),
                                  "reactiveFlow.mechanismFile must be non-empty when reactiveFlow.enabled is true");
            DNDS_assert(constMassForce.size() == 3);
            farFieldStaticValue.checkSize(nVars, "farFieldStaticValue");
            if (constMassForce.norm() || frameConstRotation.enabled ||
                std::unordered_set<EulerModel>{NS_SA, NS_SA_3D, NS_2EQ, NS_2EQ_3D}.count(model))
                DNDS_assert_info(!ignoreSourceTerm,
                                 "you have set source term, do not use ignoreSourceTerm! ");
            if (frameConstRotation.enabled)
                frameConstRotation.axis.normalize();
            for (auto &box : boxInitializers)
                box.v.checkSize(nVars, "box initial value");
            for (auto &plane : planeInitializers)
                plane.v.checkSize(nVars, "plane initial value");

            // Compute reference values
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            refU.resize(0);
            refUPrim.resize(0);
            if (reactiveFlow.enabled)
                return; // requires PhysicsProperties/ChemicalSource resolution in EulerEvaluator
            if (StateValue::filled(farFieldStaticValue.cons))
                refU = farFieldStaticValue.cons;
            else if (farFieldStaticValue.originType == StateValueOrigin::ConsSensible &&
                     StateValue::filled(farFieldStaticValue.consSensible))
                refU = farFieldStaticValue.consSensible;
            else
                return;
            refUPrim = refU;
            Gas::IdealGasThermalConservative2Primitive<dim>(refU, refUPrim, idealGasProperty.gamma, 0 /* config, sensible ρE */);
            DNDS_assert(refUPrim(I4) > 0 && refUPrim(0) > 0);
            real a = std::sqrt(idealGasProperty.gamma * refUPrim(I4) / (refUPrim(0) + verySmallReal));
            refU(Seq123).setConstant(refU(Seq123).norm() + a);
            refUPrim(Seq123).setConstant(refUPrim(Seq123).norm());
        }
    };
}
