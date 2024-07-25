#pragma once

// #ifndef __DNDS_REALLY_COMPILING__
// #define __DNDS_REALLY_COMPILING__
// #define __DNDS_REALLY_COMPILING__HEADER_ON__
// #endif

#include "Gas.hpp"
#include "Geom/Mesh.hpp"
#include "CFV/VariationalReconstruction.hpp"
#include "CFV/VariationalReconstruction_Reconstruction.hxx"
#include "CFV/VariationalReconstruction_LimiterProcedure.hxx"
#include "DNDS/JsonUtil.hpp"
#include "Euler.hpp"
#include "EulerBC.hpp"
#include "EulerJacobian.hpp"
#include "EulerEvaluatorSettings.hpp"
#include "DNDS/SerializerBase.hpp"

// #ifdef __DNDS_REALLY_COMPILING__HEADER_ON__
// #undef __DNDS_REALLY_COMPILING__
// #endif

#define JSON_ASSERT DNDS_assert
#include "json.hpp"
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

    template <EulerModel model = NS>
    class EulerEvaluator
    {
    public:
        static const int nVarsFixed = getnVarsFixed(model);
        static const int dim = getDim_Fixed(model);
        static const int gDim = getGeomDim_Fixed(model);
        static const auto I4 = dim + 1;

        typedef Eigen::VectorFMTSafe<real, dim> TVec;
        typedef Eigen::MatrixFMTSafe<real, dim, dim> TMat;
        typedef Eigen::VectorFMTSafe<real, nVarsFixed> TU;
        typedef Eigen::MatrixFMTSafe<real, nVarsFixed, nVarsFixed> TJacobianU;
        typedef Eigen::MatrixFMTSafe<real, dim, nVarsFixed> TDiffU;
        typedef Eigen::MatrixFMTSafe<real, nVarsFixed, dim> TDiffUTransposed;
        typedef ArrayDOFV<nVarsFixed> TDof;
        typedef ArrayRECV<nVarsFixed> TRec;
        typedef ArrayRECV<1> TScalar;

    public:
        // static const int gdim = 2; //* geometry dim

    private:
        int nVars = 5;

        bool passiveDiscardSource = false;

    public:
        void setPassiveDiscardSource(bool n) { passiveDiscardSource = n; }

    private:
    public:
        ssp<Geom::UnstructuredMesh> mesh;
        ssp<CFV::VariationalReconstruction<gDim>> vfv; //! gDim -> 3 for intellisense //!tmptmp
        ssp<BoundaryHandler<model>> pBCHandler;
        int kAv = 0;

        // buffer for fdtau
        std::vector<real> lambdaCell;
        std::vector<real> lambdaFace;
        std::vector<real> lambdaFaceC;
        std::vector<real> lambdaFaceVis;
        std::vector<real> deltaLambdaFace;

        std::vector<Eigen::Vector<real, Eigen::Dynamic>> dWall;
        std::vector<real> dWallFace;

        // maps from bc id to various objects
        std::map<Geom::t_index, AnchorPointRecorder<nVarsFixed>> anchorRecorders;
        std::map<Geom::t_index, OneDimProfile<nVarsFixed>> profileRecorders;
        std::map<Geom::t_index, IntegrationRecorder> bndIntegrations;
        std::map<Geom::t_index, std::ofstream> bndIntegrationLogs;

        // ArrayVDOF<25> dRdUrec;
        // ArrayVDOF<25> dRdb;

        Eigen::Vector<real, -1> fluxWallSum;
        std::vector<Eigen::Vector<real, nVarsFixed>> fluxBnd;
        index nFaceReducedOrder = 0;

        ssp<Direct::SerialSymLUStructure> symLU;

        EulerEvaluatorSettings<model> settings;

        EulerEvaluator(const decltype(mesh) &Nmesh, const decltype(vfv) &Nvfv, const decltype(pBCHandler) &npBCHandler, const decltype(settings.jsonSettings) &nJsonSettings)
            : mesh(Nmesh), vfv(Nvfv), pBCHandler(npBCHandler), kAv(Nvfv->settings.maxOrder + 1)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            nVars = getNVars(model); //! // TODO: dynamic setting it

            this->settings.jsonSettings = nJsonSettings;
            this->settings.ReadWriteJSON(settings.jsonSettings, nVars, true);

            lambdaCell.resize(mesh->NumCellProc()); // but only dist part are used, ghost part to not judge for it in facial iter
            lambdaFace.resize(mesh->NumFaceProc());
            lambdaFaceC.resize(mesh->NumFaceProc());
            lambdaFaceVis.resize(lambdaFace.size());
            deltaLambdaFace.resize(lambdaFace.size());

            fluxBnd.resize(mesh->NumBnd());
            for (auto &v : fluxBnd)
                v.resize(nVars);

            this->GetWallDist();

            if (model == NS_2EQ || model == NS_2EQ_3D)
            {
                TU farPrim = settings.farFieldStaticValue;
                real gamma = settings.idealGasProperty.gamma;
                Gas::IdealGasThermalConservative2Primitive(settings.farFieldStaticValue, farPrim, gamma);
                real T = farPrim(I4) / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * farPrim(0));
                // auto [rhs0, rhs] = RANS::SolveZeroGradEquilibrium<dim>(settings.farFieldStaticValue, this->muEff(settings.farFieldStaticValue, T));
                // if(mesh->getMPI().rank == 0)
                //     log()
                //     << "EulerEvaluator===EulerEvaluator: got 2EQ init for farFieldStaticValue: " << settings.farFieldStaticValue.transpose() << "\n"
                //     << fmt::format(" [{:.3e} -> {:.3e}] ", rhs0, rhs) << std::endl;
            }

            DNDS_MAKE_SSP(symLU, mesh->getMPI(), mesh->NumCell());
        }

        void GetWallDist();

        /******************************************************/
        void EvaluateDt(
            ArrayDOFV<1> &dt,
            ArrayDOFV<nVarsFixed> &u,
            ArrayRECV<nVarsFixed> &uRec,
            real CFL, real &dtMinall, real MaxDt = 1,
            bool UseLocaldt = false);

        static const uint64_t RHS_Ignore_Viscosity = 0x1ull;
        /**
         * @brief
         * \param rhs overwritten;
         *
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
            uint64_t flags = 0);

        void LUSGSMatrixInit(
            JacobianDiagBlock<nVarsFixed> &JDiag,
            JacobianDiagBlock<nVarsFixed> &JSource,
            ArrayDOFV<1> &dTau, real dt, real alphaDiag,
            ArrayDOFV<nVarsFixed> &u,
            ArrayRECV<nVarsFixed> &uRec,
            int jacobianCode,
            real t);

        void LUSGSMatrixVec(
            real alphaDiag,
            ArrayDOFV<nVarsFixed> &u,
            ArrayDOFV<nVarsFixed> &uInc,
            JacobianDiagBlock<nVarsFixed> &JDiag,
            ArrayDOFV<nVarsFixed> &AuInc);

        void LUSGSMatrixToJacobianLU(
            real alphaDiag,
            ArrayDOFV<nVarsFixed> &u,
            JacobianDiagBlock<nVarsFixed> &JDiag,
            JacobianLocalLU<nVarsFixed> &jacLU);

        /**
         * @brief to use LUSGS, use LUSGSForward(..., uInc, uInc); uInc.pull; LUSGSBackward(..., uInc, uInc);
         * the underlying logic is that for index, ghost > dist, so the forward uses no ghost,
         * and ghost should be pulled before using backward;
         * to use Jacobian instead of LUSGS, use LUSGSForward(..., uInc, uIncNew); LUSGSBackward(..., uInc, uIncNew); uIncNew.pull; uInc = uIncNew;
         * \param uIncNew overwritten;
         *
         */
        void UpdateLUSGSForward(
            real alphaDiag,
            ArrayDOFV<nVarsFixed> &rhs,
            ArrayDOFV<nVarsFixed> &u,
            ArrayDOFV<nVarsFixed> &uInc,
            JacobianDiagBlock<nVarsFixed> &JDiag,
            ArrayDOFV<nVarsFixed> &uIncNew);

        /**
         * @brief
         * \param uIncNew overwritten;
         *
         */
        void UpdateLUSGSBackward(
            real alphaDiag,
            ArrayDOFV<nVarsFixed> &rhs,
            ArrayDOFV<nVarsFixed> &u,
            ArrayDOFV<nVarsFixed> &uInc,
            JacobianDiagBlock<nVarsFixed> &JDiag,
            ArrayDOFV<nVarsFixed> &uIncNew);

        void UpdateSGS(
            real alphaDiag,
            ArrayDOFV<nVarsFixed> &rhs,
            ArrayDOFV<nVarsFixed> &u,
            ArrayDOFV<nVarsFixed> &uInc,
            ArrayDOFV<nVarsFixed> &uIncNew,
            JacobianDiagBlock<nVarsFixed> &JDiag,
            bool forward, TU &sumInc);

        void LUSGSMatrixSolveJacobianLU(
            real alphaDiag,
            ArrayDOFV<nVarsFixed> &rhs,
            ArrayDOFV<nVarsFixed> &u,
            ArrayDOFV<nVarsFixed> &uInc,
            ArrayDOFV<nVarsFixed> &uIncNew,
            ArrayDOFV<nVarsFixed> &bBuf,
            JacobianDiagBlock<nVarsFixed> &JDiag,
            JacobianLocalLU<nVarsFixed> &jacLU,
            TU &sumInc);

        void UpdateSGSWithRec(
            real alphaDiag,
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

        void FixUMaxFilter(ArrayDOFV<nVarsFixed> &u);

        void TimeAverageAddition(ArrayDOFV<nVarsFixed> &w, ArrayDOFV<nVarsFixed> &wAveraged, real dt, real &tCur);

        void MeanValueCons2Prim(ArrayDOFV<nVarsFixed> &u, ArrayDOFV<nVarsFixed> &w);
        void MeanValuePrim2Cons(ArrayDOFV<nVarsFixed> &w, ArrayDOFV<nVarsFixed> &u);

        using tFCompareField = std::function<TU(const Geom::tPoint &, real)>;
        using tFCompareFieldWeight = std::function<real(const Geom::tPoint &, real)>;

        void EvaluateNorm(Eigen::Vector<real, -1> &res, ArrayDOFV<nVarsFixed> &rhs, index P = 1, bool volWise = false, bool average = false);

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

        void EvaluateURecBeta(
            ArrayDOFV<nVarsFixed> &u,
            ArrayRECV<nVarsFixed> &uRec,
            ArrayDOFV<1> &uRecBeta, index &nLim, real &betaMin, int flag);

        bool AssertMeanValuePP(
            ArrayDOFV<nVarsFixed> &u, bool panic);

        /**
         * @param res is incremental residual
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
         * @param res is incremental residual fixed previously
         * @param cellRHSAlpha is limiting factor evaluated previously
         */
        void EvaluateCellRHSAlphaExpansion(
            ArrayDOFV<nVarsFixed> &u,
            ArrayRECV<nVarsFixed> &uRec,
            ArrayDOFV<1> &uRecBeta,
            ArrayDOFV<nVarsFixed> &res,
            ArrayDOFV<1> &cellRHSAlpha, index &nLim, real alphaMin);

        void MinSmoothDTau(
            ArrayDOFV<1> &dTau, ArrayDOFV<1> &dTauNew);

        /******************************************************/

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

        void UFromOtherCell(TU &u, index iFace, index iCell, index iCellOther, rowsize if2c)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            auto faceID = mesh->GetFaceZone(iFace);
            mesh->CellOtherCellPeriodicHandle(
                iFace, if2c,
                [&]()
                { u(Seq123) = mesh->periodicInfo.TransVector(Eigen::Vector<real, dim>{u(Seq123)}, faceID); },
                [&]()
                { u(Seq123) = mesh->periodicInfo.TransVectorBack(Eigen::Vector<real, dim>{u(Seq123)}, faceID); });
        }

        void DiffUFromCell2Face(TDiffU &u, index iFace, index iCell, rowsize if2c)
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
            {
                u(Seq012, Eigen::all) = mesh->periodicInfo.TransVectorBack<dim, nVarsFixed>(u(Seq012, Eigen::all), faceID);
                u(Eigen::all, Seq123) = mesh->periodicInfo.TransVectorBack<dim, dim>(u(Eigen::all, Seq123).transpose(), faceID).transpose();
            }
            if (if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID))
            {
                u(Seq012, Eigen::all) = mesh->periodicInfo.TransVector<dim, nVarsFixed>(u(Seq012, Eigen::all), faceID);
                u(Eigen::all, Seq123) = mesh->periodicInfo.TransVector<dim, dim>(u(Eigen::all, Seq123).transpose(), faceID).transpose();
            }
        }

        TU fluxFace(
            const TU &ULxy,
            const TU &URxy,
            const TU &ULMeanXy,
            const TU &URMeanXy,
            const TDiffU &DiffUxy,
            const TDiffU &DiffUxyPrim,
            const TVec &unitNorm,
            const TVec &vg,
            const TMat &normBase,
            TU &FLfix,
            TU &FRfix,
            Geom::t_index btype,
            typename Gas::RiemannSolverType rsType,
            index iFace, int ig);

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

            if constexpr (model == NS_SA || model == NS_SA_3D)
            {
                subFdU(5, 5) = un;
                subFdU(5, 0) = -un * U(5) / U(0);
                subFdU(5, 1) = n(0) * U(5) / U(0);
                subFdU(5, 2) = n(1) * U(5) / U(0);
                subFdU(5, 3) = n(2) * U(5) / U(0);
            }
            if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
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
         *
         */
        TU fluxJacobian0_Right_Times_du(
            const TU &U,
            const TVec &n,
            const TVec &vg,
            Geom::t_index btype,
            const TU &dU, real lambdaMain, real lambdaC)
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
            dF(Seq01234) -= lambdaMain * dU(Seq01234);
            if constexpr (model == NS_SA || model == NS_SA_3D)
            {
                dF(I4 + 1) = dU(I4 + 1) * n.dot(velo - vg) + U(I4 + 1) * n.dot(dVelo);
                dF(I4 + 1) -= dU(I4 + 1) * lambdaMain;
            }
            if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
            {
                dF(I4 + 1) = dU(I4 + 1) * n.dot(velo - vg) + U(I4 + 1) * n.dot(dVelo);
                dF(I4 + 1) -= dU(I4 + 1) * lambdaMain;
                dF(I4 + 2) = dU(I4 + 2) * n.dot(velo - vg) + U(I4 + 2) * n.dot(dVelo);
                dF(I4 + 2) -= dU(I4 + 2) * lambdaMain;
            }
            return dF;
        }

        TJacobianU fluxJacobian0_Right_Times_du_AsMatrix(
            const TU &U,
            const TVec &n,
            const TVec &vg,
            Geom::t_index btype,
            real lambdaMain, real lambdaC)
        { // TODO: optimize this
            TJacobianU J;
            J.resize(nVars, nVars);
            J.setIdentity();
            for (int i = 0; i < nVars; i++)
                J(Eigen::all, i) = fluxJacobian0_Right_Times_du(U, n, vg, btype, J(Eigen::all, i), lambdaMain, lambdaC);
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
            if constexpr (model == NS_SA || model == NS_SA_3D)
            {
                dF(I4 + 1) = dU(I4 + 1) * n.dot(velo - vg) + U(I4 + 1) * n.dot(dVelo);
            }
            if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
            {
                dF(I4 + 1) = dU(I4 + 1) * n.dot(velo - vg) + U(I4 + 1) * n.dot(dVelo);
                dF(I4 + 2) = dU(I4 + 2) * n.dot(velo - vg) + U(I4 + 2) * n.dot(dVelo);
            }
            return dF;
        }

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

        void TransformVelocityRotatingFrame(TU &U, const Geom::tPoint &pPhysics, int direction)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            U(Seq123) += direction * settings.frameConstRotation.vOmega().cross(pPhysics - settings.frameConstRotation.center)(Seq012) * U(0);
        }

        void TransformURotatingFrame(TU &U, const Geom::tPoint &pPhysics, int direction)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
#ifndef USE_ABS_VELO_IN_ROTATION
            U(I4) -= U(Seq123).squaredNorm() / (2 * U(0));
            U(Seq123) += direction * settings.frameConstRotation.vOmega().cross(pPhysics - settings.frameConstRotation.center)(Seq012) * U(0);
            U(I4) += U(Seq123).squaredNorm() / (2 * U(0));
#endif
        }

        void TransformURotatingFrame_ABS_VELO(TU &U, const Geom::tPoint &pPhysics, int direction)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
#ifdef USE_ABS_VELO_IN_ROTATION
            U(I4) -= U(Seq123).squaredNorm() / (2 * U(0));
            U(Seq123) += direction * settings.frameConstRotation.vOmega().cross(pPhysics - settings.frameConstRotation.center)(Seq012) * U(0);
            U(I4) += U(Seq123).squaredNorm() / (2 * U(0));
#endif
        }

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
                index iFace = mesh->bnd2face.at(iBnd);
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

        void updateBCProfiles(ArrayDOFV<nVarsFixed> &u, ArrayRECV<nVarsFixed> &uRec);

        void updateBCProfilesPressureRadialEq();

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
            int geomMode = 0);

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
            // if constexpr (model == NS_SA || model == NS_SA_3D)
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
            if constexpr (model == NS_SA || model == NS_SA_3D)
                if (ret(I4 + 1) < 0)
                    ret(I4 + 1) = umean(I4 + 1), compressed = true;
#endif
            if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
            {
                if (ret(I4 + 1) < 0)
                    ret(I4 + 1) = umean(I4 + 1), compressed = true;
                if (ret(I4 + 2) < 0)
                    ret(I4 + 2) = umean(I4 + 2), compressed = true;
            }

            return ret;
        }

        inline TU CompressInc(
            const TU &u,
            const TU &uInc)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            real rhoEps = smallReal * settings.refUPrim(0);
            real pEps = smallReal * settings.refUPrim(I4);

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
#ifndef USE_NS_SA_ALLOW_NEGATIVE_MEAN
            if constexpr (model == NS_SA || model == NS_SA_3D)
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
            if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
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

        void FixIncrement(
            ArrayDOFV<nVarsFixed> &cx,
            ArrayDOFV<nVarsFixed> &cxInc, real alpha = 1.0)
        {
            for (index iCell = 0; iCell < cxInc.Size(); iCell++)
                cxInc[iCell] = this->CompressInc(cx[iCell], cxInc[iCell] * alpha);
        }

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
                //             if (pBCHandler->GetTypeFromID(mesh->GetFaceZone(f)) == BCWall)
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

        void CentralSmoothResidual(ArrayDOFV<nVarsFixed> &r, ArrayDOFV<nVarsFixed> &rs, ArrayDOFV<nVarsFixed> &rtemp)
        {
            for (int iterS = 1; iterS <= settings.nCentralSmoothStep; iterS++)
            {
                real epsC = 0.5;
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

        void InitializeUDOF(ArrayDOFV<nVarsFixed> &u);

        struct OutputOverlapDataRefs
        {
            ArrayDOFV<nVarsFixed> &u;
            ArrayRECV<nVarsFixed> &uRec;
            ArrayDOFV<1> &betaPP;
            ArrayDOFV<1> &alphaPP;
        };

        void InitializeOutputPicker(OutputPicker &op, OutputOverlapDataRefs dataRefs);
    };
}