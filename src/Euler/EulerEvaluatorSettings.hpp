#pragma once
#include "DNDS/JsonUtil.hpp"
#include "DNDS/ConfigParam.hpp"

#include "Euler.hpp"
#include "Gas.hpp"
#include "CLDriver.hpp"
#include <unordered_set>

namespace DNDS::Euler
{
    template <EulerModel model>
    struct EulerEvaluatorSettings
    {
        static const int nVarsFixed = getnVarsFixed(model);
        static const int dim = getDim_Fixed(model);
        static const int gDim = getGeomDim_Fixed(model);
        static const auto I4 = dim + 1;

        bool useScalarJacobian = false;
        bool useRoeJacobian = false;
        bool noRsOnWall = false;
        bool noGRPOnWall = false;
        bool ignoreSourceTerm = false;

        int direct2ndRecMethod = 1;

        int specialBuiltinInitializer = 0;

        real uRecAlphaCompressPower = 1;
        real uRecBetaCompressPower = 1;
        bool forceVolURecBeta = true;
        bool ppEpsIsRelaxed = false;

        real RANSBottomLimit = 0.01;

        Gas::RiemannSolverType rsType = Gas::Roe;
        Gas::RiemannSolverType rsTypeAux = Gas::UnknownRS;
        Gas::RiemannSolverType rsTypeWall = Gas::UnknownRS;
        real rsFixScale = 1;
        real rsIncFScale = 1;
        int rsMeanValueEig = 0;
        int rsRotateScheme = 0;
        real minWallDist = 1e-12;
        int wallDistExection = 0; // 1 is serial
        real wallDistRefineMax = 1;
        int wallDistScheme = 0;
        int wallDistCellLoadSize = 1024 * 32;
        int wallDistIter = 1000;
        int wallDistLinSolver = 0; // 0 for jacobi, 1 for gmres
        real wallDistResTol = 1e-4;
        int wallDistIterStart = 100;
        int wallDistPoissonP = 2;
        real wallDistDTauScale = 100.;
        int wallDistNJacobiSweep = 10;
        real SADESScale = veryLargeReal;
        int SADESMode = 1;
        RANSModel ransModel = RANSModel::RANS_None;
        int ransUseQCR = 0;
        int ransSARotCorrection = 1;
        int ransEigScheme = 0;
        int ransForce2nd = 0;
        int ransSource2nd = 0;
        int source2nd = 0;
        int usePrimGradInVisFlux = 0;
        int useSourceGradFixGG = 0;
        int nCentralSmoothStep = 0;
        real centralSmoothEps = 0.5;
        Eigen::Vector<real, 3> constMassForce = Eigen::Vector<real, 3>{0, 0, 0};
        struct FrameConstRotation
        {
            bool enabled = false;
            Geom::tPoint axis = Geom::tPoint{0, 0, 1};
            Geom::tPoint center = Geom::tPoint{0, 0, 0};
            real rpm = 0;
            real Omega()
            {
                return rpm * (2 * pi / 60.);
            }
            Geom::tPoint vOmega()
            {
                return axis * Omega();
            }
            Geom::tPoint rVec(const Geom::tPoint &r)
            {
                return r - r.dot(axis) * axis;
            }
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
        } frameConstRotation;
        CLDriverSettings cLDriverSettings;
        std::vector<std::string> cLDriverBCNames;
        Eigen::Vector<real, -1> farFieldStaticValue = Eigen::Vector<real, 5>{1, 0, 0, 0, 2.5};
        struct BoxInitializer
        {
            real x0{0}, x1{0}, y0{0}, y1{0}, z0{0}, z1{0};
            Eigen::Vector<real, -1> v;

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
        std::vector<BoxInitializer> boxInitializers;

        struct PlaneInitializer
        {
            real a{0}, b{0}, c{0}, h{0};
            Eigen::Vector<real, -1> v;

            DNDS_DECLARE_CONFIG(PlaneInitializer)
            {
                DNDS_FIELD(a, "Plane normal x-component");
                DNDS_FIELD(b, "Plane normal y-component");
                DNDS_FIELD(c, "Plane normal z-component");
                DNDS_FIELD(h, "Plane offset");
                DNDS_FIELD(v, "Initial value vector (size = nVars)");
            }
        };
        std::vector<PlaneInitializer> planeInitializers;

        struct ExprtkInitializer
        {
            std::vector<std::string> exprs;

            DNDS_DECLARE_CONFIG(ExprtkInitializer)
            {
                DNDS_FIELD(exprs, "Expression lines (concatenated with newlines)");
            }

            std::string GetExpr() const
            {
                std::string ret;
                for (auto &line : exprs)
                    ret += line + "\n";
                return ret;
            }
        };
        std::vector<ExprtkInitializer> exprtkInitializers;

        struct IdealGasProperty
        {
            real gamma = 1.4;
            real Rgas = 1;
            real muGas = 1;
            real prGas = 0.72;
            real CpGas = Rgas * gamma / (gamma - 1); // derived, not serialized
            real TRef = 273.15;
            real CSutherland = 110.4;
            int muModel = 1; // 0=constant, 1=sutherland, 2=constant_nu

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

            /// Recompute derived quantities after deserialization.
            void recomputeDerived()
            {
                CpGas = Rgas * gamma / (gamma - 1);
            }
        } idealGasProperty;

        /***************************************************************************************************/
        // end of setting entries
        /***************************************************************************************************/

        Eigen::Vector<real, -1> refU;
        Eigen::Vector<real, -1> refUPrim;

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
        }

        EulerEvaluatorSettings() = default;

        EulerEvaluatorSettings(int nVars)
        {
            if constexpr (model == NS_SA || model == NS_SA_3D)
            {
                ransModel = RANSModel::RANS_SA;
            }
            if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
            {
                ransModel = RANSModel::RANS_KOWilcox;
            }
            farFieldStaticValue.setOnes(nVars);
            DNDS_assert(nVars > I4);
            farFieldStaticValue(0) = 1;
            farFieldStaticValue(Eigen::seq(Eigen::fix<0>, Eigen::fix<I4>)).setZero();
            farFieldStaticValue(I4) = 2.5;
        }

        /// @brief Post-deserialization finalization: semantic checks and derived quantities.
        /// Must be called after from_json (or typed copy) with the runtime nVars.
        void finalize(int nVars)
        {
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