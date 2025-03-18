#pragma once
#include <nlohmann/json.hpp>

#include "Euler.hpp"
#include "Gas.hpp"
#include "DNDS/JsonUtil.hpp"
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

        nlohmann::ordered_json jsonSettings;

        bool useScalarJacobian = false;
        bool useRoeJacobian = false;
        bool noRsOnWall = false;
        bool noGRPOnWall = false;
        bool ignoreSourceTerm = false;

        int direct2ndRecMethod = 1;

        int specialBuiltinInitializer = 0;

        real uRecAlphaCompressPower = 1;
        real uRecBetaCompressPower = 11;
        bool forceVolURecBeta = true;
        bool ppEpsIsRelaxed = false;

        real RANSBottomLimit = 0.01;

        Gas::RiemannSolverType rsType = Gas::Roe;
        Gas::RiemannSolverType rsTypeAux = Gas::UnknownRS;
        Gas::RiemannSolverType rsTypeWall = Gas::UnknownRS;
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
        RANSModel ransModel = RANSModel::RANS_None;
        int ransUseQCR = 0;
        int ransSARotCorrection = 1;
        int ransEigScheme = 0;
        int ransForce2nd = 0;
        int ransSource2nd = 0;
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
                ret(Eigen::all, 0) = rn;
                ret(Eigen::all, 2) = axis;
                ret(Eigen::all, 1) = axis.cross(rn);
                return ret;
            }
            DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                FrameConstRotation,
                enabled,
                axis,
                center,
                rpm)
        } frameConstRotation;
        CLDriverSettings cLDriverSettings;
        std::vector<std::string> cLDriverBCNames;
        Eigen::Vector<real, -1> farFieldStaticValue = Eigen::Vector<real, 5>{1, 0, 0, 0, 2.5};
        struct BoxInitializer
        {
            real x0, x1, y0, y1, z0, z1;
            Eigen::Vector<real, -1> v;
            void ReadWriteJSON(nlohmann::ordered_json &jsonObj, int nVars, bool read)
            {
                __DNDS__json_to_config(x0);
                __DNDS__json_to_config(x1);
                __DNDS__json_to_config(y0);
                __DNDS__json_to_config(y1);
                __DNDS__json_to_config(z0);
                __DNDS__json_to_config(z1);
                __DNDS__json_to_config(v);
                // std::cout << "here2" << std::endl;
                if (read)
                    DNDS_assert_info(v.size() == nVars, "initial value dimension incorrect");
                // if (read)
                //     v = JsonGetEigenVector(jsonObj["v"]), DNDS_assert(v.size() == nVars);
                // else
                //     jsonObj["v"] = EigenVectorGetJson(v);
            }
        };
        std::vector<BoxInitializer> boxInitializers;

        struct PlaneInitializer
        {
            real a, b, c, h;
            Eigen::Vector<real, -1> v;
            void ReadWriteJSON(nlohmann::ordered_json &jsonObj, int nVars, bool read)
            {
                __DNDS__json_to_config(a);
                __DNDS__json_to_config(b);
                __DNDS__json_to_config(c);
                __DNDS__json_to_config(h);
                __DNDS__json_to_config(v);
                if (read)
                    DNDS_assert_info(v.size() == nVars, "initial value dimension incorrect");
                // if (read)
                //     v = JsonGetEigenVector(jsonObj["v"]), DNDS_assert(v.size() == nVars);
                // else
                //     jsonObj["v"] = EigenVectorGetJson(v);
            }
        };
        std::vector<PlaneInitializer> planeInitializers;

        struct ExprtkInitializer
        {
            std::vector<std::string> exprs;
            DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                ExprtkInitializer,
                exprs)
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
            real CpGas = Rgas * gamma / (gamma - 1);
            real TRef = 273.15;
            real CSutherland = 110.4;
            int muModel = 1; // 0=constant, 1=sutherland, 2=constant_nu

            void ReadWriteJSON(nlohmann::ordered_json &jsonObj, bool read)
            {
                __DNDS__json_to_config(gamma);
                __DNDS__json_to_config(Rgas);
                __DNDS__json_to_config(muGas);
                __DNDS__json_to_config(prGas);
                __DNDS__json_to_config(TRef);
                __DNDS__json_to_config(CSutherland);
                __DNDS__json_to_config(muModel);
                CpGas = Rgas * gamma / (gamma - 1);
            }
        } idealGasProperty;

        /***************************************************************************************************/
        // end of setting entries
        /***************************************************************************************************/

        Eigen::Vector<real, -1> refU;
        Eigen::Vector<real, -1> refUPrim;

        EulerEvaluatorSettings()
        {
            if constexpr (model == NS_SA || model == NS_SA_3D)
            {
                ransModel = RANSModel::RANS_SA;
            }
            if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
            {
                ransModel = RANSModel::RANS_KOWilcox;
            }
        }

        void ReadWriteJSON(nlohmann::ordered_json &jsonObj, int nVars, bool read)
        {

            //********* root entries

            __DNDS__json_to_config(useScalarJacobian);
            __DNDS__json_to_config(useRoeJacobian);
            if (read)
            {
                DNDS_assert(!(useScalarJacobian && useRoeJacobian));
            }
            __DNDS__json_to_config(noRsOnWall);
            __DNDS__json_to_config(noGRPOnWall);
            __DNDS__json_to_config(ignoreSourceTerm);
            __DNDS__json_to_config(direct2ndRecMethod);
            __DNDS__json_to_config(specialBuiltinInitializer);
            __DNDS__json_to_config(uRecAlphaCompressPower);
            __DNDS__json_to_config(uRecBetaCompressPower);
            __DNDS__json_to_config(forceVolURecBeta);
            __DNDS__json_to_config(ppEpsIsRelaxed);
            __DNDS__json_to_config(RANSBottomLimit);
            Gas::RiemannSolverType riemannSolverType = rsType;
            __DNDS__json_to_config(riemannSolverType);
            rsType = riemannSolverType;
            Gas::RiemannSolverType riemannSolverTypeAux = rsTypeAux;
            __DNDS__json_to_config(riemannSolverTypeAux);
            rsTypeAux = riemannSolverTypeAux;
            Gas::RiemannSolverType riemannSolverTypeWall = rsTypeWall;
            __DNDS__json_to_config(riemannSolverTypeWall);
            rsTypeWall = riemannSolverTypeWall;
            // std::cout << rsType << std::endl;
            __DNDS__json_to_config(rsMeanValueEig);
            __DNDS__json_to_config(rsRotateScheme);
            __DNDS__json_to_config(minWallDist);
            __DNDS__json_to_config(wallDistExection);
            __DNDS__json_to_config(wallDistRefineMax);
            __DNDS__json_to_config(wallDistScheme);
            __DNDS__json_to_config(wallDistCellLoadSize);
            __DNDS__json_to_config(wallDistIter);
            __DNDS__json_to_config(wallDistLinSolver);
            __DNDS__json_to_config(wallDistResTol);
            __DNDS__json_to_config(wallDistIterStart);
            __DNDS__json_to_config(wallDistPoissonP);
            __DNDS__json_to_config(wallDistDTauScale);
            __DNDS__json_to_config(wallDistNJacobiSweep);
            __DNDS__json_to_config(SADESScale);
            __DNDS__json_to_config(ransModel);
            __DNDS__json_to_config(ransUseQCR);
            __DNDS__json_to_config(ransSARotCorrection);
            __DNDS__json_to_config(ransEigScheme);
            __DNDS__json_to_config(ransForce2nd);
            __DNDS__json_to_config(ransSource2nd);
            __DNDS__json_to_config(usePrimGradInVisFlux);
            __DNDS__json_to_config(useSourceGradFixGG);
            __DNDS__json_to_config(nCentralSmoothStep);
            __DNDS__json_to_config(centralSmoothEps);
            __DNDS__json_to_config(constMassForce);
            __DNDS__json_to_config(frameConstRotation);
            __DNDS__json_to_config(cLDriverSettings);
            __DNDS__json_to_config(cLDriverBCNames);
            if (read)
                DNDS_assert(constMassForce.size() == 3);
            __DNDS__json_to_config(farFieldStaticValue);
            if (read)
                DNDS_assert(farFieldStaticValue.size() == nVars);
            if (read)
                DNDS_assert(ransModel != RANS_Unknown);

            //********* box entries
            if (read)
            {
                boxInitializers.clear();
                DNDS_assert(jsonObj["boxInitializers"].is_array());
                for (auto &boxJson : jsonObj["boxInitializers"])
                {
                    DNDS_assert(boxJson.is_object());
                    BoxInitializer box;
                    box.ReadWriteJSON(boxJson, nVars, read);
                    boxInitializers.push_back(box);
                }
                if (constMassForce.norm() || frameConstRotation.enabled ||
                    std::unordered_set<EulerModel>{NS_SA, NS_SA_3D, NS_2EQ, NS_2EQ_3D}.count(model))
                    DNDS_assert_info(!ignoreSourceTerm, "you have set source term, do not use ignoreSourceTerm! ");
                if (frameConstRotation.enabled)
                    frameConstRotation.axis.normalize();
            }
            else
            {
                jsonObj["boxInitializers"] = nlohmann::ordered_json::array();
                for (auto &b : boxInitializers)
                {
                    nlohmann::ordered_json j;
                    b.ReadWriteJSON(j, nVars, read);
                    jsonObj["boxInitializers"].push_back(j);
                }
            }

            //********* plane entries
            if (read)
            {
                planeInitializers.clear();
                DNDS_assert(jsonObj["planeInitializers"].is_array());
                for (auto &planeJson : jsonObj["planeInitializers"])
                {
                    DNDS_assert(planeJson.is_object());
                    PlaneInitializer p;
                    p.ReadWriteJSON(planeJson, nVars, read);
                    planeInitializers.push_back(p);
                }
            }
            else
            {
                jsonObj["planeInitializers"] = nlohmann::ordered_json::array();
                for (auto &p : planeInitializers)
                {
                    nlohmann::ordered_json j;
                    p.ReadWriteJSON(j, nVars, read);
                    jsonObj["planeInitializers"].push_back(j);
                }
            }

            __DNDS__json_to_config(exprtkInitializers);

            //********* idealGasProperty
            if (read)
                DNDS_assert(jsonObj["idealGasProperty"].is_object());
            idealGasProperty.ReadWriteJSON(jsonObj["idealGasProperty"], read);

            if (read)
            {
                DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
                refU = farFieldStaticValue;
                refUPrim = refU;
                Gas::IdealGasThermalConservative2Primitive<dim>(refU, refUPrim, idealGasProperty.gamma);
                DNDS_assert(refUPrim(I4) > 0 && refUPrim(0) > 0);
                real a = std::sqrt(idealGasProperty.gamma * refUPrim(I4) / (refUPrim(0) + verySmallReal));
                refU(Seq123).setConstant(refU(Seq123).norm() + a);
                refUPrim(Seq123).setConstant(refUPrim(Seq123).norm());
            }
        }
    };
}