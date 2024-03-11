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
#include "RANS_ke.hpp"
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
        /**
         * @brief
         * \param rhs overwritten;
         *
         */
        void EvaluateRHS(
            ArrayDOFV<nVarsFixed> &rhs,
            ArrayDOFV<nVarsFixed> &JSource,
            ArrayDOFV<nVarsFixed> &u,
            ArrayRECV<nVarsFixed> &uRec,
            ArrayDOFV<1> &uRecBeta,
            ArrayDOFV<1> &cellRHSAlpha,
            bool onlyOnHalfAlpha,
            real t);

        void LUSGSMatrixInit(
            ArrayDOFV<nVarsFixed> &JDiag,
            ArrayDOFV<nVarsFixed> &JSource,
            ArrayDOFV<1> &dTau, real dt, real alphaDiag,
            ArrayDOFV<nVarsFixed> &u,
            ArrayRECV<nVarsFixed> &uRec,
            int jacobianCode,
            real t);

        void LUSGSMatrixVec(
            real alphaDiag,
            ArrayDOFV<nVarsFixed> &u,
            ArrayDOFV<nVarsFixed> &uInc,
            ArrayDOFV<nVarsFixed> &JDiag,
            ArrayDOFV<nVarsFixed> &AuInc);

        void LUSGSMatrixToJacobianLU(
            real alphaDiag,
            ArrayDOFV<nVarsFixed> &u,
            ArrayDOFV<nVarsFixed> &JDiag,
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
            ArrayDOFV<nVarsFixed> &JDiag,
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
            ArrayDOFV<nVarsFixed> &JDiag,
            ArrayDOFV<nVarsFixed> &uIncNew);

        void UpdateSGS(
            real alphaDiag,
            ArrayDOFV<nVarsFixed> &rhs,
            ArrayDOFV<nVarsFixed> &u,
            ArrayDOFV<nVarsFixed> &uInc,
            ArrayDOFV<nVarsFixed> &uIncNew,
            ArrayDOFV<nVarsFixed> &JDiag,
            bool forward, TU &sumInc);

        void LUSGSMatrixSolveJacobianLU(
            real alphaDiag,
            ArrayDOFV<nVarsFixed> &rhs,
            ArrayDOFV<nVarsFixed> &u,
            ArrayDOFV<nVarsFixed> &uInc,
            ArrayDOFV<nVarsFixed> &uIncNew,
            ArrayDOFV<nVarsFixed> &bBuf,
            ArrayDOFV<nVarsFixed> &JDiag,
            JacobianLocalLU<nVarsFixed> &jacLU,
            TU &sumInc);

        void UpdateSGSWithRec(
            real alphaDiag,
            ArrayDOFV<nVarsFixed> &rhs,
            ArrayDOFV<nVarsFixed> &u,
            ArrayRECV<nVarsFixed> &uRec,
            ArrayDOFV<nVarsFixed> &uInc,
            ArrayRECV<nVarsFixed> &uRecInc,
            ArrayDOFV<nVarsFixed> &JDiag,
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

        void EvaluateNorm(Eigen::Vector<real, -1> &res, ArrayDOFV<nVarsFixed> &rhs, index P = 1, bool volWise = false);

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
            real relax,
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
            if (!mesh->isPeriodic)
                return;
            auto faceID = mesh->GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return;
            if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicDonor(faceID))) // I am donor
                u(Seq123) = mesh->periodicInfo.TransVector(Eigen::Vector<real, dim>{u(Seq123)}, faceID);
            if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicMain(faceID))) // I am main
                u(Seq123) = mesh->periodicInfo.TransVectorBack(Eigen::Vector<real, dim>{u(Seq123)}, faceID);
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
            const TVec &unitNorm,
            const TMat &normBase,
            TU &FLfix,
            TU &FRfix,
            Geom::t_index btype,
            typename Gas::RiemannSolverType rsType,
            index iFace, int ig);

        TU source(
            const TU &UMeanXy,
            const TDiffU &DiffUxy,
            index iCell, index ig)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
#ifdef DNDS_FV_EULEREVALUATOR_SOURCE_TERM_ZERO
            TU ret;
            ret.resizeLike(UMeanXy);
            ret.setZero();
            return ret;
#endif
            if constexpr (model == NS || model == NS_2D || model == NS_3D)
            {
                TU ret;
                ret.resizeLike(UMeanXy);
                ret.setZero();
                ret(Seq123) = settings.constMassForce(Seq012) * UMeanXy(0);
                return ret;
            }
            else if constexpr (model == NS_SA || model == NS_SA_3D)
            {
                real d = std::min(dWall[iCell][ig], std::pow(veryLargeReal, 1. / 6.));
                real cb1 = 0.1355;
                real cb2 = 0.622;
                real sigma = 2. / 3.;
                real cnu1 = 7.1;
                real cnu2 = 0.7;
                real cnu3 = 0.9;
                real cw2 = 0.3;
                real cw3 = 2;
                real kappa = 0.41;
                real rlim = 10;
                real cw1 = cb1 / sqr(kappa) + (1 + cb2) / sigma;

                real ct3 = 1.2;
                real ct4 = 0.5;

                real pMean, asqrMean, Hmean;
                real gamma = settings.idealGasProperty.gamma;
                Gas::IdealGasThermal(UMeanXy(I4), UMeanXy(0), (UMeanXy(Seq123) / UMeanXy(0)).squaredNorm(),
                                     gamma, pMean, asqrMean, Hmean);
                // ! refvalue:
                real muRef = settings.idealGasProperty.muGas;

                real nuh = UMeanXy(I4 + 1) * muRef / UMeanXy(0);

                real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * UMeanXy(0));
                real mufPhy, muf;
                mufPhy = muf = muEff(UMeanXy, T);

                real Chi = (UMeanXy(I4 + 1) * muRef / mufPhy);
                real fnu1 = std::pow(Chi, 3) / (std::pow(Chi, 3) + std::pow(cnu1, 3));
                real fnu2 = 1 - Chi / (1 + Chi * fnu1);

                Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
                Eigen::Matrix<real, dim, 1> diffRhoNu = DiffUxy(Seq012, {I4 + 1}) * muRef;
                Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
                Eigen::Matrix<real, dim, 1> diffNu = (diffRhoNu - nuh * diffRho) / UMeanXy(0);
                Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
                Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);

                Eigen::Matrix<real, dim, dim> Omega = 0.5 * (diffU.transpose() - diffU);
                real S = Omega.norm() * std::sqrt(2);
                real Sbar = nuh / (sqr(kappa) * sqr(d)) * fnu2;

                real Sh;
#ifdef USE_NS_SA_NEGATIVE_MODEL
                if (Sbar < -cnu2 * S)
                    Sh = S + S * (sqr(cnu2) * S + cnu3 * Sbar) / ((cnu3 - 2 * cnu2) * S - Sbar);
                else //*negative fix
#endif
                    Sh = S + Sbar;

                real r = std::min(nuh / (Sh * sqr(kappa * d) + verySmallReal), rlim);
                real g = r + cw2 * (std::pow(r, 6) - r);
                real fw = g * std::pow((1 + std::pow(cw3, 6)) / (std::pow(g, 6) + std::pow(cw3, 6)), 1. / 6.);

                real ft2 = ct3 * std::exp(-ct4 * sqr(Chi));
#ifdef USE_NS_SA_NEGATIVE_MODEL
                real D = (cw1 * fw - cb1 / sqr(kappa) * ft2) * sqr(nuh / d); //! modified >>
                real P = cb1 * (1 - ft2) * Sh * nuh;                         //! modified >>
#else
                real D = (cw1 * fw - cb1 / sqr(kappa) * ft2) * sqr(nuh / d);
                real P = cb1 * (1 - ft2) * Sh * nuh;
#endif
                real fn = 1;
#ifdef USE_NS_SA_NEGATIVE_MODEL
                if (UMeanXy(I4 + 1) < 0)
                {
                    real cn1 = 16;
                    real Chi = UMeanXy(I4 + 1) * muRef / mufPhy;
                    fn = (cn1 + std::pow(Chi, 3)) / (cn1 - std::pow(Chi, 3));
                    P = cb1 * (1 - ct3) * S * nuh;
                    D = -cw1 * sqr(nuh / d);
                }
#endif

                TU ret;
                ret.resizeLike(UMeanXy);
                ret.setZero();
                ret(Seq123) = settings.constMassForce * UMeanXy(0);

                if (passiveDiscardSource)
                    P = D = 0;
                ret(I4 + 1) = UMeanXy(0) * (P - D + diffNu.squaredNorm() * cb2 / sigma) / muRef -
                              (UMeanXy(I4 + 1) * fn * muRef + mufPhy) / (UMeanXy(0) * sigma) * diffRho.dot(diffNu) / muRef;
                // std::cout << "P, D " << P / muRef << " " << D / muRef << " " << diffNu.squaredNorm() << std::endl;
                if (ret.hasNaN())
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
                // if (passiveDiscardSource)
                //     ret(Eigen::seq(5, Eigen::last)).setZero();
                return ret;
            }
            else if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
            {
                TU ret;
                ret.resizeLike(UMeanXy);
                ret.setZero();
                ret(Seq123) = settings.constMassForce * UMeanXy(0);

                real pMean, asqrMean, Hmean;
                real gamma = settings.idealGasProperty.gamma;
                Gas::IdealGasThermal(UMeanXy(I4), UMeanXy(0), (UMeanXy(Seq123) / UMeanXy(0)).squaredNorm(),
                                     gamma, pMean, asqrMean, Hmean);
                // ! refvalue:
                real muRef = settings.idealGasProperty.muGas;
                real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * UMeanXy(0));

                real mufPhy, muf;
                mufPhy = muf = muEff(UMeanXy, T);
                RANS::GetSource_RealizableKe<dim>(UMeanXy, DiffUxy, mufPhy, ret);
                return ret;
            }
            else
            {
                DNDS_assert(false);
            }
        }

        // zeroth means needs not derivative
        TU sourceJacobianDiag(
            const TU &UMeanXy,
            const TDiffU &DiffUxy,
            index iCell, index ig)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
#ifdef DNDS_FV_EULEREVALUATOR_SOURCE_TERM_ZERO
            TU ret;
            ret.resizeLike(UMeanXy);
            ret.setZero();
            return ret;
#endif
            if constexpr (model == NS || model == NS_2D || model == NS_3D)
            {
            }
            else if constexpr (model == NS_SA || model == NS_SA_3D)
            {
                real d = std::min(dWall[iCell][ig], std::pow(veryLargeReal, 1. / 6.));
                real cb1 = 0.1355;
                real cb2 = 0.622;
                real sigma = 2. / 3.;
                real cnu1 = 7.1;
                real cnu2 = 0.7;
                real cnu3 = 0.9;
                real cw2 = 0.3;
                real cw3 = 2;
                real kappa = 0.41;
                real rlim = 10;
                real cw1 = cb1 / sqr(kappa) + (1 + cb2) / sigma;

                real ct3 = 1.2;
                real ct4 = 0.5;

                real pMean, asqrMean, Hmean;
                real gamma = settings.idealGasProperty.gamma;
                Gas::IdealGasThermal(UMeanXy(I4), UMeanXy(0), (UMeanXy(Seq123) / UMeanXy(0)).squaredNorm(),
                                     gamma, pMean, asqrMean, Hmean);
                // ! refvalue:
                real muRef = settings.idealGasProperty.muGas;

                real nuh = UMeanXy(I4 + 1) * muRef / UMeanXy(0);

                real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * UMeanXy(0));
                real mufPhy, muf;
                mufPhy = muf = muEff(UMeanXy, T);

                real Chi = (UMeanXy(I4 + 1) * muRef / mufPhy);
                real fnu1 = std::pow(Chi, 3) / (std::pow(Chi, 3) + std::pow(cnu1, 3));
                real fnu2 = 1 - Chi / (1 + Chi * fnu1);

                Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
                Eigen::Matrix<real, dim, 1> diffRhoNu = DiffUxy(Seq012, {I4 + 1}) * muRef;
                Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
                Eigen::Matrix<real, dim, 1> diffNu = (diffRhoNu - nuh * diffRho) / UMeanXy(0);
                Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
                Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);

                Eigen::Matrix<real, dim, dim> Omega = 0.5 * (diffU.transpose() - diffU);
                real S = Omega.norm() * std::sqrt(2);
                real Sbar = nuh / (sqr(kappa) * sqr(d)) * fnu2;

                real Sh;
#ifdef USE_NS_SA_NEGATIVE_MODEL
                if (Sbar < -cnu2 * S)
                    Sh = S + S * (sqr(cnu2) * S + cnu3 * Sbar) / ((cnu3 - 2 * cnu2) * S - Sbar);
                else
#endif
                    Sh = S + Sbar;

                real r = std::min(nuh / (Sh * sqr(kappa * d) + verySmallReal), rlim);
                real g = r + cw2 * (std::pow(r, 6) - r);
                real fw = g * std::pow((1 + std::pow(cw3, 6)) / (std::pow(g, 6) + std::pow(cw3, 6)), 1. / 6.);

                real ft2 = ct3 * std::exp(-ct4 * sqr(Chi));
                real D = (cw1 * fw - cb1 / sqr(kappa) * ft2) * sqr(nuh / d); //! modified >>
                real P = cb1 * (1 - ft2) * Sh * nuh;                         //! modified >>
                // real D = cw1 * fw * sqr(nuh / d);
                // real P = cb1 * Sh * nuh;
                real fn = 1;
#ifdef USE_NS_SA_NEGATIVE_MODEL
                if (UMeanXy(I4 + 1) < 0)
                {
                    real cn1 = 16;
                    real Chi = UMeanXy(I4 + 1) * muRef / mufPhy;
                    fn = (cn1 + std::pow(Chi, 3)) / (cn1 - std::pow(Chi, 3));
                    P = cb1 * (1 - ct3) * S * nuh;
                    D = -cw1 * sqr(nuh / d);
                }
#endif

                TU ret;
                ret.resizeLike(UMeanXy);
                ret.setZero();

                if (passiveDiscardSource)
                    P = D = 0;
                ret(I4 + 1) = -std::min(UMeanXy(0) * (P * 0 - D * 2) / muRef / (UMeanXy(I4 + 1) + verySmallReal), -verySmallReal);
                // std::cout << ret(I4+1) << std::endl;

                if (ret.hasNaN())
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
                    std::cout << UMeanXy.transpose() << std::endl;
                    std::cout << pMean << std::endl;
                    std::cout << ret.transpose() << std::endl;

                    DNDS_assert(false);
                }
                // if (passiveDiscardSource)
                //     ret(Eigen::seq(5, Eigen::last)).setZero();
                return ret;
            }
            else if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
            {
                TU ret;
                ret.resizeLike(UMeanXy);
                ret.setZero();

                real pMean, asqrMean, Hmean;
                real gamma = settings.idealGasProperty.gamma;
                Gas::IdealGasThermal(UMeanXy(I4), UMeanXy(0), (UMeanXy(Seq123) / UMeanXy(0)).squaredNorm(),
                                     gamma, pMean, asqrMean, Hmean);
                // ! refvalue:
                real muRef = settings.idealGasProperty.muGas;
                real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * UMeanXy(0));

                real mufPhy, muf;
                mufPhy = muf = muEff(UMeanXy, T);
                RANS::GetSourceJacobianDiag_RealizableKe<dim>(UMeanXy, DiffUxy, mufPhy, ret);
                return ret;
            }
            else
            {
                DNDS_assert(false);
            }

            TU Ret;
            Ret.resizeLike(UMeanXy);
            Ret.setZero();
            return Ret;
        }

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
                velo, dVelo,
                dp, p,
                dF);
            dF(Seq01234) -= lambdaMain * dU(Seq01234);
            if constexpr (model == NS_SA || model == NS_SA_3D)
            {
                dF(I4 + 1) = dU(I4 + 1) * n.dot(velo) + U(I4 + 1) * n.dot(dVelo);
                dF(I4 + 1) -= dU(I4 + 1) * lambdaMain;
            }
            if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
            {
                dF(I4 + 1) = dU(I4 + 1) * n.dot(velo) + U(I4 + 1) * n.dot(dVelo);
                dF(I4 + 1) -= dU(I4 + 1) * lambdaMain;
                dF(I4 + 2) = dU(I4 + 2) * n.dot(velo) + U(I4 + 2) * n.dot(dVelo);
                dF(I4 + 2) -= dU(I4 + 2) * lambdaMain;
            }
            return dF;
        }

        TJacobianU fluxJacobian0_Right_Times_du_AsMatrix(
            const TU &U,
            const TVec &n,
            Geom::t_index btype,
            real lambdaMain, real lambdaC)
        { // TODO: optimize this
            TJacobianU J;
            J.resize(nVars, nVars);
            J.setIdentity();
            for (int i = 0; i < nVars; i++)
                J(Eigen::all, i) = fluxJacobian0_Right_Times_du(U, n, btype, J(Eigen::all, i), lambdaMain, lambdaC);
            return J;
        }

        TU fluxJacobianC_Right_Times_du(
            const TU &U,
            const TVec &n,
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
                velo, dVelo,
                dp, p,
                dF);
            if constexpr (model == NS_SA || model == NS_SA_3D)
            {
                dF(I4 + 1) = dU(I4 + 1) * n.dot(velo) + U(I4 + 1) * n.dot(dVelo);
            }
            if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
            {
                dF(I4 + 1) = dU(I4 + 1) * n.dot(velo) + U(I4 + 1) * n.dot(dVelo);
                dF(I4 + 2) = dU(I4 + 2) * n.dot(velo) + U(I4 + 2) * n.dot(dVelo);
            }
            return dF;
        }

        TU generateBoundaryValue(
            TU &ULxy, //! warning, possible that UL is also modified
            const TU &ULMeanXy,
            index iCell, index iFace,
            const TVec &uNorm,
            const TMat &normBase,
            const TVec &pPhysics,
            real t,
            Geom::t_index btype,
            bool fixUL = false)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS

            TU URxy;
            URxy.resizeLike(ULxy);

            if (btype == Geom::BC_ID_DEFAULT_FAR ||
                btype == Geom::BC_ID_DEFAULT_SPECIAL_DMR_FAR ||
                btype == Geom::BC_ID_DEFAULT_SPECIAL_RT_FAR ||
                btype == Geom::BC_ID_DEFAULT_SPECIAL_IV_FAR ||
                btype == Geom::BC_ID_DEFAULT_SPECIAL_2DRiemann_FAR ||
                pBCHandler->GetTypeFromID(btype) == EulerBCType::BCFar)
            {
                DNDS_assert(ULxy(0) > 0);
                if (btype == Geom::BC_ID_DEFAULT_FAR ||
                    pBCHandler->GetTypeFromID(btype) == EulerBCType::BCFar)
                {
                    const TU &far = btype >= Geom::BC_ID_DEFAULT_MAX
                                        ? pBCHandler->GetValueFromID(btype)
                                        : TU(settings.farFieldStaticValue);
                    // fmt::print("far id: {}\n", btype);
                    // std::cout << far.transpose() << std::endl;

                    real un = ULxy(Seq123).dot(uNorm) / ULxy(0);
                    real vsqr = (ULxy(Seq123) / ULxy(0)).squaredNorm();
                    real gamma = settings.idealGasProperty.gamma;
                    real asqr, H, p;
                    Gas::IdealGasThermal(ULxy(I4), ULxy(0), vsqr, gamma, p, asqr, H);

                    DNDS_assert(asqr >= 0);
                    real a = std::sqrt(asqr);

                    if (un - a > 0) // full outflow
                    {
                        URxy = ULxy;
                    }
                    else if (un > 0) //  1 sonic outflow, 1 sonic inflow, other outflow (subsonic out)
                    {
                        TU farPrimitive, ULxyPrimitive;
                        farPrimitive.resizeLike(ULxy);
                        ULxyPrimitive.resizeLike(URxy);
                        Gas::IdealGasThermalConservative2Primitive<dim>(far, farPrimitive, gamma);
                        Gas::IdealGasThermalConservative2Primitive<dim>(ULxy, ULxyPrimitive, gamma);
                        ULxyPrimitive(I4) = farPrimitive(I4); // using far pressure
                        Gas::IdealGasThermalPrimitive2Conservative<dim>(ULxyPrimitive, URxy, gamma);
                    }
                    else if (un + a > 0) //  1 sonic outflow, 1 sonic inflow, other inflow (subsonic in)
                    {
                        TU farPrimitive, ULxyPrimitive;
                        farPrimitive.resizeLike(ULxy);
                        ULxyPrimitive.resizeLike(URxy);
                        Gas::IdealGasThermalConservative2Primitive<dim>(far, farPrimitive, gamma);
                        Gas::IdealGasThermalConservative2Primitive<dim>(ULxy, ULxyPrimitive, gamma);
                        // farPrimitive(0) = ULxyPrimitive(0); // using inner density
                        farPrimitive(I4) = ULxyPrimitive(I4); // using inner pressure
                        Gas::IdealGasThermalPrimitive2Conservative<dim>(farPrimitive, URxy, gamma);
                    }
                    else // full inflow
                    {
                        URxy = far;
                    }
                    // URxy = far; //! override
                }
                else if (btype == Geom::BC_ID_DEFAULT_SPECIAL_DMR_FAR)
                {
                    DNDS_assert(dim > 1);
                    URxy = settings.farFieldStaticValue;
                    real uShock = 10;
                    if constexpr (dim == 3) //* manual static dispatch
                    {
                        if (((pPhysics(0) - uShock / std::sin(pi / 3) * t - 1. / 6.) -
                             pPhysics(1) / std::tan(pi / 3)) > 0)
                            URxy({0, 1, 2, 3, 4}) = Eigen::Vector<real, 5>{1.4, 0, 0, 0, 2.5};
                        else
                            URxy({0, 1, 2, 3, 4}) = Eigen::Vector<real, 5>{8, 57.157676649772960, -33, 0, 5.635e2};
                    }
                    else
                    {
                        if (((pPhysics(0) - uShock / std::sin(pi / 3) * t - 1. / 6.) -
                             pPhysics(1) / std::tan(pi / 3)) > 0)
                            URxy({0, 1, 2, 3}) = Eigen::Vector<real, 4>{1.4, 0, 0, 2.5};
                        else
                            URxy({0, 1, 2, 3}) = Eigen::Vector<real, 4>{8, 57.157676649772960, -33, 5.635e2};
                    }
                }
                else if (btype == Geom::BC_ID_DEFAULT_SPECIAL_RT_FAR)
                {
                    DNDS_assert(dim > 1);
                    Eigen::VectorXd far = settings.farFieldStaticValue;
                    real gamma = settings.idealGasProperty.gamma;
                    real un = ULxy(Seq123).dot(uNorm) / ULxy(0);
                    real vsqr = (ULxy(Seq123) / ULxy(0)).squaredNorm();
                    real asqr, H, p;
                    Gas::IdealGasThermal(ULxy(I4), ULxy(0), vsqr, gamma, p, asqr, H);

                    DNDS_assert(asqr >= 0);
                    real a = std::sqrt(asqr);
                    real v = -0.025 * a * cos(pPhysics(0) * 8 * pi);

                    if (pPhysics(1) < 0.5)
                    {

                        real rho = 2;
                        real p = 1;
                        far(0) = rho;
                        far(1) = 0;
                        far(2) = rho * v;
                        far(I4) = 0.5 * rho * sqr(v) + p / (gamma - 1);
                    }
                    else
                    {
                        real rho = 1;
                        real p = 2.5;
                        far(0) = rho;
                        far(1) = 0;
                        far(2) = rho * v;
                        far(I4) = 0.5 * rho * sqr(v) + p / (gamma - 1);
                    }

                    if (un - a > 0) // full outflow
                    {
                        URxy = ULxy;
                    }
                    else if (un > 0) //  1 sonic outflow, 1 sonic inflow, other outflow (subsonic out)
                    {
                        TU farPrimitive, ULxyPrimitive;
                        farPrimitive.resizeLike(ULxy);
                        ULxyPrimitive.resizeLike(URxy);
                        Gas::IdealGasThermalConservative2Primitive<dim>(far, farPrimitive, gamma);
                        Gas::IdealGasThermalConservative2Primitive<dim>(ULxy, ULxyPrimitive, gamma);
                        ULxyPrimitive(I4) = farPrimitive(I4); // using far pressure
                        Gas::IdealGasThermalPrimitive2Conservative<dim>(ULxyPrimitive, URxy, gamma);
                    }
                    else if (un + a > 0) //  1 sonic outflow, 1 sonic inflow, other inflow (subsonic in)
                    {
                        TU farPrimitive, ULxyPrimitive;
                        farPrimitive.resizeLike(ULxy);
                        ULxyPrimitive.resizeLike(URxy);
                        Gas::IdealGasThermalConservative2Primitive<dim>(far, farPrimitive, gamma);
                        Gas::IdealGasThermalConservative2Primitive<dim>(ULxy, ULxyPrimitive, gamma);
                        // farPrimitive(0) = ULxyPrimitive(0); // using inner density
                        farPrimitive(I4) = ULxyPrimitive(I4); // using inner pressure
                        Gas::IdealGasThermalPrimitive2Conservative<dim>(farPrimitive, URxy, gamma);
                    }
                    else // full inflow
                    {
                        URxy = far;
                    }
                    // URxy = far; //! override
                }
                else if (btype == Geom::BC_ID_DEFAULT_SPECIAL_IV_FAR)
                {
                    real chi = 5;
                    real gamma = settings.idealGasProperty.gamma;
                    real xc = 5 + t;
                    real yc = 5 + t;
                    real r = std::sqrt(sqr(pPhysics(0) - xc) + sqr(pPhysics(1) - yc));
                    real dT = -(gamma - 1) / (8 * gamma * sqr(pi)) * sqr(chi) * std::exp(1 - sqr(r));
                    real dux = chi / 2 / pi * std::exp((1 - sqr(r)) / 2) * -(pPhysics(1) - xc);
                    real duy = chi / 2 / pi * std::exp((1 - sqr(r)) / 2) * +(pPhysics(0) - yc);
                    real T = dT + 1;
                    real ux = dux + 1;
                    real uy = duy + 1;
                    real S = 1;
                    real rho = std::pow(T / S, 1 / (gamma - 1));
                    real p = T * rho;

                    real E = 0.5 * (sqr(ux) + sqr(uy)) * rho + p / (gamma - 1);

                    // std::cout << T << " " << rho << std::endl;
                    URxy.setZero();
                    URxy(0) = rho;
                    URxy(1) = rho * ux;
                    URxy(2) = rho * uy;
                    URxy(dim + 1) = E;
                }
                else if (btype == Geom::BC_ID_DEFAULT_SPECIAL_2DRiemann_FAR)
                {
                    real gamma = settings.idealGasProperty.gamma;
                    real bdL = 0.0; // left
                    real bdR = 1.0; // right
                    real bdD = 0.0; // down
                    real bdU = 1.0; // up

                    real phi1 = -0.663324958071080;
                    real phi2 = -0.422115882408869;
                    real location = 0.8;
                    real p1 = location + phi1 * t;
                    real p2 = location + phi2 * t;
                    real rho, u, v, pre;
                    TU ULxyPrimitive;
                    ULxyPrimitive.resizeLike(ULxy);

                    Gas::IdealGasThermalConservative2Primitive<dim>(ULxy, ULxyPrimitive, gamma);
                    real rhoL = ULxyPrimitive(0);
                    real uL = ULxyPrimitive(1);
                    real vL = ULxyPrimitive(2);
                    real preL = ULxyPrimitive(I4);
                    TU farPrimitive = ULxyPrimitive;

                    static const real bTol = 1e-9;
                    if (std::abs(pPhysics(0) - bdL) < bTol)
                    { // left, phi2
                        if (pPhysics(1) <= p2)
                        { // region 3
                            rho = 0.137992831541219;
                            u = 1.206045378311055;
                            v = 1.206045378311055;
                            pre = 0.029032258064516;
                        }
                        else
                        { // region 2
                            rho = 0.532258064516129;
                            u = 1.206045378311055;
                            v = 0.0;
                            pre = 0.3;
                        }
                    }
                    else if (std::abs(pPhysics(0) - bdR) < bTol)
                    { // right, phi1
                        if (pPhysics(1) <= p1)
                        { // region 4
                            // rho = 0.532258064516129;
                            // u = 0.0;
                            // v = 1.206045378311055;
                            // pre = 0.3;
                            rho = rhoL;
                            u = -uL;
                            v = vL;
                            pre = preL;
                        }
                        else
                        { // region 1
                            // rho = 1.5;
                            // u = 0.0;
                            // v = 0.0;
                            // pre = 1.5;
                            rho = rhoL;
                            u = -uL;
                            v = vL;
                            pre = preL;
                        }
                    }
                    else if (std::abs(pPhysics(1) - bdU) < bTol)
                    { // up, phi1
                        if (pPhysics(0) <= p1)
                        { // region 2
                            // rho = 0.532258064516129;
                            // u = 1.206045378311055;
                            // v = 0.0;
                            // pre = 0.3;
                            rho = rhoL;
                            u = uL;
                            v = -vL;
                            pre = preL;
                        }
                        else
                        { // region 1
                            // rho = 1.5;
                            // u = 0.0;
                            // v = 0.0;
                            // pre = 1.5;
                            rho = rhoL;
                            u = uL;
                            v = -vL;
                            pre = preL;
                        }
                    }
                    else if (std::abs(pPhysics(1) - bdD) < bTol)
                    { // down, phi2
                        if (pPhysics(0) <= p2)
                        { // region 3
                            rho = 0.137992831541219;
                            u = 1.206045378311055;
                            v = 1.206045378311055;
                            pre = 0.029032258064516;
                        }
                        else
                        { // region 4
                            rho = 0.532258064516129;
                            u = 0.0;
                            v = 1.206045378311055;
                            pre = 0.3;
                        }
                    }
                    else
                    {
                        rho = u = v = pre = std::nan("1");
                        DNDS_assert(false); // not valid boundary pos
                    }
                    farPrimitive(0) = rho;
                    farPrimitive(1) = u, farPrimitive(2) = v;
                    farPrimitive(I4) = pre;
                    Gas::IdealGasThermalPrimitive2Conservative<dim>(farPrimitive, URxy, gamma);
                }
                else
                    DNDS_assert(false);
            }
            else if (pBCHandler->GetTypeFromID(btype) == EulerBCType::BCWallInvis ||
                     pBCHandler->GetTypeFromID(btype) == EulerBCType::BCSym)
            {
                URxy = ULxy;
                URxy(Seq123) -= 2 * ULxy(Seq123).dot(uNorm) * uNorm; // mirrored!
            }
            else if (pBCHandler->GetTypeFromID(btype) == EulerBCType::BCWall)
            {
                URxy = ULxy;
                URxy(Seq123) *= -1;
                if (model == NS_SA || model == NS_SA_3D)
                {
                    URxy(I4 + 1) *= -1;
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                    if (fixUL)
                        ULxy(I4 + 1) = URxy(I4 + 1) = 0; //! modifing UL
#endif
                }
                if (model == NS_2EQ || model == NS_2EQ_3D)
                {
                    URxy({I4 + 1, I4 + 2}) *= -1;
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                    if (fixUL)
                        ULxy({I4 + 1, I4 + 2}).setZero(), URxy({I4 + 1, I4 + 2}).setZero(); //! modifing UL
#endif
                    TVec v = (vfv->GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, -1) - vfv->GetCellBary(iCell))(Seq012);
                    real d1 = std::abs(v.dot(uNorm)); //! warning! first wall could be bad
                    real k1 = ULMeanXy(I4 + 1) / ULMeanXy(0);

                    real pMean, asqrMean, Hmean;
                    real gamma = settings.idealGasProperty.gamma;
                    Gas::IdealGasThermal(ULMeanXy(I4), ULMeanXy(0), (ULMeanXy(Seq123) / ULMeanXy(0)).squaredNorm(),
                                         gamma, pMean, asqrMean, Hmean);
                    // ! refvalue:
                    real muRef = settings.idealGasProperty.muGas;
                    real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * ULMeanXy(0));
                    real mufPhy1;
                    mufPhy1 = muEff(ULMeanXy, T);
                    real epsWall = 2 * mufPhy1 / ULMeanXy(0) * k1 / sqr(d1);
                    URxy(I4 + 2) = 2 * epsWall * ULxy(0) - ULxy(I4 + 2);
                    if (fixUL)
                        ULxy(I4 + 2) = URxy(I4 + 2) = epsWall * ULxy(0);
                    // std::cout << "d1" <<d1 << std::endl;
                }
            }
            else if (pBCHandler->GetTypeFromID(btype) == EulerBCType::BCOut)
            {
                URxy = ULxy;
            }
            else if (pBCHandler->GetTypeFromID(btype) == EulerBCType::BCIn)
            {
                URxy = pBCHandler->GetValueFromID(btype);
            }
            else
            {
                DNDS_assert(false);
            }
            return URxy;
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
                    newu5 = std::max(1e-6, newu5);
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
                    newu5 = std::max(1e-6, newu5);
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
    };
}