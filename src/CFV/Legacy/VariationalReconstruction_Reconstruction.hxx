#pragma once

#include "CFV/Legacy/VariationalReconstruction.hpp"
#include "Geom/Compat/LegacyMeshRead.hpp"

namespace DNDS
{
    namespace CFV
    {

        static const int dim = 3;
        static const int nVarsFixed = 6;

        DNDS_SWITCH_INTELLISENSE(
            template <int dim>
            template <int nVarsFixed>
            ,
            template <>
            template <>
        )
        auto VariationalReconstruction<dim>::GetBoundaryRHS(tURec<nVarsFixed> &uRec,
                                                            tUDof<nVarsFixed> &u,
                                                            index iCell, index iFace,
                                                            const TFBoundary<nVarsFixed> &FBoundary)
        {
            using namespace Geom;
            using namespace Geom::Elem;
            static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);

            Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> BCC;
            BCC.setZero(uRec[iCell].rows(), uRec[iCell].cols());
            auto faceID = mesh->GetFaceZone(iFace);
            DNDS_assert(FaceIDIsExternalBC(faceID));

            auto qFace = this->GetFaceQuad(iFace);
            if (settings.intOrderVRBCIsSame() || settings.functionalSettings.greenGauss1Weight != 0)
                qFace.IntegrationSimple(
                    BCC,
                    [&](auto &vInc, int iG)
                    {
                        RowVectorXR dbv =
                            this->GetIntPointDiffBaseValue(iCell, iFace, 0, iG, std::array<int, 1>{0}, 1);
                        Eigen::Vector<real, nVarsFixed> uBL =
                            (dbv *
                             uRec[iCell])
                                .transpose();
                        uBL += u[iCell]; //! need fixing?
                        Eigen::Vector<real, nVarsFixed> uBV =
                            FBoundary(
                                uBL,
                                u[iCell], iCell, iFace, iG,
                                this->GetFaceNorm(iFace, iG),
                                this->GetFaceQuadraturePPhysFromCell(iFace, iCell, 0, iG), faceID);
                        Eigen::RowVector<real, nVarsFixed> uIncBV = (uBV - u[iCell]).transpose();
                        if (settings.intOrderVRBCIsSame())
                            vInc = this->FFaceFunctional(dbv, uIncBV, iFace, iCell, iCell);
                        else
                            vInc.resizeLike(BCC), vInc.setZero();

                        if (settings.functionalSettings.greenGauss1Weight != 0)
                        {
                            // DNDS_assert(false); // not yet implemented
                            vInc += (settings.functionalSettings.greenGauss1Bias * this->GetGreenGauss1WeightOnCell(iCell) *
                                     this->coefficients_.matrixAHalf_GG[iCell].transpose() * this->GetFaceNorm(iFace, iG)(Seq012)) *
                                    uIncBV;
                        }
                        vInc *= this->GetFaceJacobiDet(iFace, iG);
                        // std::cout << baseWeight_.faceWeight[iFace].transpose() << std::endl;
                    });
            if (!settings.intOrderVRBCIsSame())
            {
                auto qFace = Quadrature(mesh->GetFaceElement(iFace), settings.intOrderVRBCValue());
                tSmallCoords coords;
                mesh->GetCoordsOnFace(iFace, coords);
                qFace.Integration(
                    BCC,
                    [&](auto &vInc, int __xxx_iG, const tPoint &pParam, const Elem::tD01Nj &DiNj) { // todo: cache these for bnd: pPhy JDet norm and dbv
                        BndVRPointCache &bndCacheEntry = baseWeight_.bndVRCaches.at(iFace).at(__xxx_iG);
                        auto &dbv = bndCacheEntry.D0Bj;
                        auto &np = bndCacheEntry.norm;
                        auto &JDet = bndCacheEntry.JDet;
                        auto &pPhy = bndCacheEntry.PPhy;

                        Eigen::Vector<real, nVarsFixed> uBL = (dbv * uRec[iCell]).transpose();
                        uBL += u[iCell];
                        Eigen::Vector<real, nVarsFixed> uBV =
                            FBoundary(
                                uBL,
                                u[iCell], iCell, iFace, -2,
                                np,
                                this->GetFacePointFromCell(iFace, iCell, 0, pPhy), faceID);
                        Eigen::RowVector<real, nVarsFixed> uIncBV = (uBV - u[iCell]).transpose();
                        vInc = this->FFaceFunctional(dbv, uIncBV, iFace, iCell, iCell);
                        vInc *= JDet;
                    });
            }
            return BCC;
        }

        template <int dim>
        template <int nVarsFixed>
        auto VariationalReconstruction<dim>::GetBoundaryRHSDiff(tURec<nVarsFixed> &uRec,
                                                                tURec<nVarsFixed> &uRecDiff,
                                                                tUDof<nVarsFixed> &u,
                                                                index iCell, index iFace,
                                                                const TFBoundaryDiff<nVarsFixed> &FBoundaryDiff)
        {
            using namespace Geom;
            using namespace Geom::Elem;
            static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);

            Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> BCC;
            BCC.setZero(uRec[iCell].rows(), uRec[iCell].cols());
            auto faceID = mesh->GetFaceZone(iFace);
            DNDS_assert(FaceIDIsExternalBC(faceID));

            auto qFace = this->GetFaceQuad(iFace);
            if (settings.intOrderVRBCIsSame() || settings.functionalSettings.greenGauss1Weight != 0)
                qFace.IntegrationSimple(
                    BCC,
                    [&](auto &vInc, int iG)
                    {
                        RowVectorXR dbv =
                            this->GetIntPointDiffBaseValue(iCell, iFace, 0, iG, std::array<int, 1>{0}, 1);
                        Eigen::Vector<real, nVarsFixed> uBL = (dbv * uRec[iCell]).transpose();
                        uBL += u[iCell]; //! need fixing?
                        Eigen::Vector<real, nVarsFixed> uBLDiff = (dbv * uRecDiff[iCell]).transpose();
                        Eigen::Vector<real, nVarsFixed>
                            uBV = FBoundaryDiff(
                                uBL, uBLDiff,
                                u[iCell], iCell, iFace, iG,
                                this->GetFaceNorm(iFace, iG),
                                this->GetFaceQuadraturePPhysFromCell(iFace, iCell, 0, iG), faceID);
                        Eigen::RowVector<real, nVarsFixed> uIncBV = uBV.transpose();
                        if (settings.intOrderVRBCIsSame())
                            vInc = this->FFaceFunctional(dbv, uIncBV, iFace, iCell, iCell);
                        else
                            vInc.resizeLike(BCC), vInc.setZero();
                        // std::cout << baseWeight_.faceWeight[iFace].transpose() << std::endl;
                        if (settings.functionalSettings.greenGauss1Weight != 0)
                        {
                            // DNDS_assert(false); // not yet implemented
                            vInc += (settings.functionalSettings.greenGauss1Bias * this->GetGreenGauss1WeightOnCell(iCell) *
                                     this->coefficients_.matrixAHalf_GG[iCell].transpose() * this->GetFaceNorm(iFace, iG)(Seq012)) *
                                    uIncBV;
                        }
                        vInc *= this->GetFaceJacobiDet(iFace, iG);
                    });
            if (!settings.intOrderVRBCIsSame())
            {
                auto qFace = Quadrature(mesh->GetFaceElement(iFace), settings.intOrderVRBCValue());
                tSmallCoords coords;
                mesh->GetCoordsOnFace(iFace, coords);
                qFace.Integration(
                    BCC,
                    [&](auto &vInc, int __xxx_iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                    {
                        BndVRPointCache &bndCacheEntry = baseWeight_.bndVRCaches.at(iFace).at(__xxx_iG);
                        auto &dbv = bndCacheEntry.D0Bj;
                        auto &np = bndCacheEntry.norm;
                        auto &JDet = bndCacheEntry.JDet;
                        auto &pPhy = bndCacheEntry.PPhy;
                        Eigen::Vector<real, nVarsFixed> uBL = (dbv * uRec[iCell]).transpose();
                        uBL += u[iCell];
                        Eigen::Vector<real, nVarsFixed> uBLDiff = (dbv * uRecDiff[iCell]).transpose();
                        Eigen::Vector<real, nVarsFixed> uBV =
                            FBoundaryDiff(
                                uBL, uBLDiff,
                                u[iCell], iCell, iFace, -2,
                                np,
                                this->GetFacePointFromCell(iFace, iCell, 0, pPhy), faceID);
                        Eigen::RowVector<real, nVarsFixed> uIncBV = uBV.transpose();
                        vInc = this->FFaceFunctional(dbv, uIncBV, iFace, iCell, iCell);
                        vInc *= JDet;
                    });
            }
            return BCC;
        }

        DNDS_SWITCH_INTELLISENSE(
            template <int dim>
            template <int nVarsFixed>
            ,
            template <>
            template <>
        )
        void VariationalReconstruction<dim>::DoReconstruction2ndGrad(
            tUGrad<nVarsFixed, dim> &uRec,
            tUDof<nVarsFixed> &u,
            const TFBoundary<nVarsFixed> &FBoundary,
            int method,
            ReconstructionMaskView reconstructionMask)
        {
            auto meshRead = this->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
            const auto meshLegacy = meshRead.LegacyHostView();
            const auto *mesh = &meshLegacy;
            this->ValidateMeshDependencies();
            using namespace Geom;
            using namespace Geom::Elem;

            static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>); // note this is gDim!
            static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);

            if (method == 1 || method == 11)
            {
                // simple gauss rule
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    int nVars = u[iCell].size();
                    auto c2f = mesh->cell2face[iCell];
                    Eigen::Matrix<real, nVarsFixed, dim> grad;
                    grad.setZero(nVars, dim);
                    MatrixXR mGG;
                    Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> bGG;
                    if (method == 11)
                    {
                        mGG.setZero(dim + c2f.size(), dim + c2f.size());
                        bGG.setZero(dim + c2f.size(), u[0].rows());
                        mGG(Seq012, Seq012).setIdentity(); // for GGMP
                        // mGG(Seq012, Seq012) *= this->GetCellVol(iCell);
                    }

                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                    {
                        index iFace = c2f[ic2f];
                        index iCellOther = mesh->CellFaceOther(iCell, iFace, ic2f);
                        auto faceID = mesh->GetFaceZone(iFace);
                        int if2c = mesh->CellIsFaceBack(iCell, iFace, ic2f) ? 0 : 1;
                        auto faceCent = this->GetFaceQuadraturePPhysFromCell(iFace, iCell, if2c, -1);
                        real lThis = (faceCent - this->GetCellBary(iCell)).norm();
                        if (settings.subs2ndOrderGGScheme == 0)
                            lThis = 1;
                        real lThat = lThis;
                        if (iCellOther != UnInitIndex)
                        {
                            lThat = (this->GetOtherCellBaryFromCell(iCell, iCellOther, iFace, if2c) - faceCent).norm();
                            if (settings.subs2ndOrderGGScheme == 0)
                                lThat = 1;
                            Eigen::RowVector<real, nVarsFixed> uOther = u[iCellOther].transpose();
                            this->ApplyPeriodicTransform(if2c, faceID, uOther);
                            Eigen::Vector<real, dim> uNorm = this->GetFaceNormFromCell(iFace, iCell, if2c, -1)(Seq012) * (if2c ? -1. : 1.);
                            Eigen::Matrix<real, nVarsFixed, dim> gradInc =
                                (uOther.transpose() - u[iCell]) *
                                lThis / (lThis + lThat + verySmallReal) * this->GetFaceArea(iFace) * uNorm.transpose();
                            grad += gradInc;
                            if (method == 11)
                            {
                                mGG(Seq012, dim + ic2f) = -this->GetFaceArea(iFace) / (this->GetCellVol(iCell) + verySmallReal) * uNorm;
                                mGG(dim + ic2f, Seq012) =
                                    (this->GetCellBary(iCell)(Seq012) + this->GetOtherCellBaryFromCell(iCell, iCellOther, iFace, if2c)(Seq012) -
                                     2 * faceCent(Seq012))
                                        .transpose();
                                mGG(dim + ic2f, dim + ic2f) = 2;
                                bGG(dim + ic2f, EigenAll) = uOther + u[iCell].transpose();
                            }
                        }
                        else
                        {
                            auto faceID = mesh->GetFaceZone(iFace);
                            DNDS_assert(FaceIDIsExternalBC(faceID));

                            Eigen::Vector<real, nVarsFixed> uBL = u[iCell]; //! need fixing?
                            Eigen::Vector<real, nVarsFixed> uBV =
                                FBoundary(
                                    uBL,
                                    u[iCell], iCell, iFace, -1,
                                    this->GetFaceNorm(iFace, -1),
                                    faceCent, faceID);
                            Eigen::Vector<real, dim> uNorm = this->GetFaceNorm(iFace, -1)(Seq012);
                            grad += (uBV - u[iCell]) * 0.5 * this->GetFaceArea(iFace) * uNorm.transpose();
                            if (method == 11)
                            {
                                mGG(Seq012, dim + ic2f) = -this->GetFaceArea(iFace) / (this->GetCellVol(iCell) + verySmallReal) * uNorm;
                                // Eigen::Vector<real, dim> BaryOther = this->GetCellBary(iCell)(Seq012) +
                                //                                      2 * uNorm * uNorm.dot(faceCent(Seq012) - this->GetCellBary(iCell)(Seq012));
                                Eigen::Vector<real, dim> BaryOther = 2 * faceCent(Seq012) - this->GetCellBary(iCell)(Seq012);
                                mGG(dim + ic2f, Seq012) = (this->GetCellBary(iCell)(Seq012) + BaryOther -
                                                           2 * faceCent(Seq012))
                                                              .transpose();
                                mGG(dim + ic2f, dim + ic2f) = 2;
                                bGG(dim + ic2f, EigenAll) = (uBV + u[iCell]).transpose();
                            }
                        }
                    }

                    grad /= GetCellVol(iCell);

                    if (method == 11)
                    {

                        // auto mGGLU = mGG.colPivHouseholderQr();
                        auto mGGLU = mGG.fullPivLu();
                        DNDS_assert(mGGLU.isInvertible());
                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> xGG = mGGLU.solve(bGG);
                        grad = xGG(Seq012, EigenAll).transpose();

                        if (mesh->GetCellElement(iCell).type == Elem::Pyramid5)
                        {
                            std::cout << "pyramid5\n";
                            std::cout << mGG << std::endl;
                            std::cout << bGG << std::endl;
                            auto svd = mGG.bdcSvd();
                            std::cout << svd.singularValues().transpose() << std::endl;
                            std::cout << svd.singularValues().maxCoeff() / svd.singularValues().minCoeff() << std::endl;
                        }
                        if (mesh->GetCellElement(iCell).type == Elem::Tet4)
                        {
                            std::cout << "Tet4\n";
                            std::cout << mGG << std::endl;
                            std::cout << bGG << std::endl;
                            auto svd = mGG.bdcSvd();
                            std::cout << svd.singularValues().transpose() << std::endl;
                            std::cout << svd.singularValues().maxCoeff() / svd.singularValues().minCoeff() << std::endl;
                        }
                    }

                    // tPoint cellBary = GetCellBary(iCell);
                    // Eigen::MatrixXd vvv = (grad * (Geom::RotZ(90) * cellBary)(Eigen::seq(0, dim - 1))).transpose();
                    // std::cout << cellBary.transpose() << " ---- " << vvv << std::endl;
                    // DNDS_assert(vvv(0) < 1e-10);

                    uRec[iCell] = grad.transpose();
                }
            }
            else if (method == 2) //! warning, periodic not implemented here
            {
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    int nVars = u[iCell].size();
                    auto c2f = mesh->cell2face[iCell];
                    Eigen::Matrix<real, nVarsFixed, dim> grad;
                    grad.setZero(nVars, dim);
                    Eigen::Matrix<real, Eigen::Dynamic, dim> dcs;
                    Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> dus;
                    dcs.resize(c2f.size(), dim);
                    dus.resize(c2f.size(), nVars);

                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                    {
                        index iFace = c2f[ic2f];
                        index iCellOther = mesh->CellFaceOther(iCell, iFace, ic2f);
                        int if2c = mesh->CellIsFaceBack(iCell, iFace, ic2f) ? 0 : 1;

                        if (iCellOther != UnInitIndex)
                        {
                            dcs(ic2f, EigenAll) = (GetOtherCellBaryFromCell(iCell, iCellOther, iFace, if2c) - GetCellBary(iCell))(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>))
                                                      .transpose();
                            dus(ic2f, EigenAll) = (u[iCellOther] - u[iCell]).transpose();
                        }
                        else
                        {
                            auto faceID = mesh->GetFaceZone(iFace);
                            DNDS_assert(FaceIDIsExternalBC(faceID));

                            Eigen::Vector<real, nVarsFixed> uBL;
                            uBL.setZero();
                            uBL += u[iCell]; //! need fixing?
                            Eigen::Vector<real, nVarsFixed> uBV =
                                FBoundary(
                                    uBL,
                                    u[iCell], iCell, iFace, -1,
                                    this->GetFaceNorm(iFace, -1),
                                    this->GetFaceQuadraturePPhysFromCell(iFace, iCell, if2c, -1), faceID);
                            dus(ic2f, EigenAll) = (uBV - u[iCell]).transpose();

                            dcs(ic2f, EigenAll) =
                                ((GetCellBary(iCell) - GetFaceQuadraturePPhys(iFace, -1)).dot(GetFaceNorm(iFace, -1)) * (-2.) * GetFaceNorm(iFace, -1))(
                                    Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>))
                                    .transpose();
                        }
                    }
                    // Eigen::MatrixXd m;
                    // m.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve()
                    auto svd = dcs.template bdcSvd<Eigen::ComputeFullU | Eigen::ComputeFullV>();
                    grad = svd.solve(dus).transpose();
                    uRec[iCell] = grad.transpose();
                }
            }
            else
            {
                DNDS_assert_info(false, "NO SUCH 2nd rec METHOD");
            }
            reconstructionMask.Apply(uRec);
        }

        DNDS_SWITCH_INTELLISENSE(
            template <int dim>
            template <int nVarsFixed>
            ,
            template <>
            template <>
        )
        void VariationalReconstruction<dim>::ConvertUGradToURec(
            tURec<nVarsFixed> &uRec,
            tUGrad<nVarsFixed, dim> &uGrad,
            const std::vector<int> &mask,
            ReconstructionMaskView reconstructionMask)
        {
            auto meshRead = this->mesh->AcquireRead(Geom::MeshReadRequirements::Primary());
            const auto meshLegacy = meshRead.LegacyHostView();
            const auto *mesh = &meshLegacy;
            this->ValidateMeshDependencies();
            using namespace Geom;
            using namespace Geom::Elem;
            static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>); // note this is gDim!
            static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)

            // #if defined(DNDS_DIST_MT_USE_OMP)
            // #pragma omp parallel for schedule(static)
            // #endif
            //             for (int iPart = 0; iPart < mesh->NLocalParts(); iPart++)
            //                 for (index iCell = mesh->LocalPartStart(iPart); iCell < mesh->LocalPartEnd(iPart); iCell++)
            {
                Eigen::Matrix<real, nVarsFixed, dim> grad = uGrad[iCell].transpose();

                Eigen::Matrix<real, dim, dim> d1bv;
                d1bv = this->GetIntPointDiffBaseValue(
                    iCell, -1, -1, -1, Seq123, dim + 1)(EigenAll, Seq012);
                auto lud1bv = d1bv.partialPivLu();
                if (lud1bv.rcond() > 1e9)
                    std::cout << "Large Cond " << lud1bv.rcond() << std::endl;
                if (mask.size() == 0)
                {
                    uRec[iCell].setZero();
                    uRec[iCell](Seq012, EigenAll) = lud1bv.solve(grad.transpose());
                }
                else
                {
                    uRec[iCell](EigenAll, mask).setZero();
                    uRec[iCell](Seq012, mask) = lud1bv.solve(grad.transpose())(EigenAll, mask);
                }
            }
            reconstructionMask.Apply(uRec);
        }

        DNDS_SWITCH_INTELLISENSE(
            template <int dim>
            template <int nVarsFixed>
            ,
            template <>
            template <>
        )
        void VariationalReconstruction<dim>::DoReconstruction2nd(
            tURec<nVarsFixed> &uRec,
            tUDof<nVarsFixed> &u,
            const TFBoundary<nVarsFixed> &FBoundary,
            int method,
            const std::vector<int> &mask,
            ReconstructionMaskView reconstructionMask)
        {
            tUGrad<nVarsFixed, dim> uGrad;
            this->BuildUGrad(uGrad, u.father->MatRowSize(), false, false);
            this->DoReconstruction2ndGrad(uGrad, u, FBoundary, method, reconstructionMask);
            this->ConvertUGradToURec(uRec, uGrad, mask, reconstructionMask);
        }

        template <int dim>
        template <int nVarsFixed>
        void VariationalReconstruction<dim>::DoReconstructionIter(
            tURec<nVarsFixed> &uRec,
            tURec<nVarsFixed> &uRecNew,
            tUDof<nVarsFixed> &u,
            const TFBoundary<nVarsFixed> &FBoundary,
            bool putIntoNew,
            bool recordInc,
            bool uRecIsZero,
            ReconstructionMaskView reconstructionMask)
        {
            DoReconstructionIterInternal<nVarsFixed>(
                uRec, uRecNew, u, FBoundary, nullptr, nullptr,
                putIntoNew, recordInc, uRecIsZero, reconstructionMask);
        }

        template <int dim>
        template <int nVarsFixed>
        void VariationalReconstruction<dim>::DoReconstructionIterLimited(
            tURec<nVarsFixed> &uRec,
            tURec<nVarsFixed> &uRecNew,
            tUDof<nVarsFixed> &u,
            const TFBoundary<nVarsFixed> &FBoundary,
            tURec<nVarsFixed> &uRecTarget,
            tUDof<1> &alpha,
            bool putIntoNew,
            ReconstructionMaskView reconstructionMask)
        {
            DoReconstructionIterInternal<nVarsFixed>(
                uRec, uRecNew, u, FBoundary, &uRecTarget, &alpha,
                putIntoNew, false, false, reconstructionMask);
        }

        template <int dim>
        template <int nVarsFixed>
        void VariationalReconstruction<dim>::DoReconstructionIterInternal(
            tURec<nVarsFixed> &uRec,
            tURec<nVarsFixed> &uRecNew,
            tUDof<nVarsFixed> &u,
            const TFBoundary<nVarsFixed> &FBoundary,
            tURec<nVarsFixed> *uRecTarget,
            tUDof<1> *alpha,
            bool putIntoNew,
            bool recordInc,
            bool uRecIsZero,
            ReconstructionMaskView reconstructionMask)
        {
            using namespace Geom;
            using namespace Geom::Elem;
            using namespace Geom::Base;
            static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
            int maxNDOF = GetNDof<dim>(settings.maxOrder);
            DNDS_assert_info((uRecTarget == nullptr) == (alpha == nullptr),
                             "uRecTarget and alpha must be supplied together");
            DNDS_assert_info(uRecTarget == nullptr || !recordInc,
                             "limited reconstruction does not support increment recording");
            DNDS_assert_info(uRecTarget == nullptr || settings.maxOrder > 1,
                             "limited reconstruction requires maxOrder > 1");
            if (recordInc)
                DNDS_assert_info(putIntoNew, "the -RHS must be put into uRecNew");
            if (!recordInc)
            {
                reconstructionMask.Apply(uRec);    // Also clear old/restarted ghost coefficients before a sweep.
                reconstructionMask.Apply(uRecNew); // Either array may receive the update, including its ghost storage.
            }
            if (settings.maxOrder == 1 && settings.subs2ndOrder != 0)
            {
                if (recordInc)
                    this->DoReconstruction2nd<nVarsFixed>(uRecNew, u, (FBoundary), settings.subs2ndOrder, std::vector<int>(), reconstructionMask);
                else if (putIntoNew)
                    this->DoReconstruction2nd<nVarsFixed>(uRecNew, u, (FBoundary), settings.subs2ndOrder, std::vector<int>(), reconstructionMask);
                else
                    this->DoReconstruction2nd<nVarsFixed>(uRec, u, (FBoundary), settings.subs2ndOrder, std::vector<int>(), reconstructionMask);
                if (recordInc && !uRecIsZero && reconstructionMask.selectedCells)
                    for (index iCell : *reconstructionMask.selectedCells)
                        if (iCell < mesh->NumCell())
                            uRecNew[iCell] = uRec[iCell];
                return;
            }

            auto meshRead = this->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
            const auto meshLegacy = meshRead.LegacyHostView();
            const auto *mesh = &meshLegacy;
            this->ValidateMeshDependencies();
            // #if defined(DNDS_DIST_MT_USE_OMP)
            // #pragma omp parallel for schedule(static)
            // #endif
            //             for (index iCell = 0; iCell < mesh->NumCell(); iCell++)

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (int iPart = 0; iPart < this->mesh->NLocalParts(); iPart++)
                for (const auto range : this->mesh->LocalPartRanges(iPart).Intervals())
                    for (index iCell = range.begin; iCell < range.end; ++iCell)
                    {
                        if (reconstructionMask.IsMasked(iCell))
                        {
                            // The constrained equation is uRec_i=0. Its residual is
                            // uRec_i, not a zero row (Krylov operators stay nonsingular).
                            if (recordInc && !uRecIsZero)
                                uRecNew[iCell] = uRec[iCell];
                            else
                                uRecNew[iCell].setZero();
                            continue;
                        }
                        real relax = GetCellAtr(iCell).relax;
                        real alphaCell = alpha == nullptr ? 0.0 : (*alpha)[iCell](0);
                        DNDS_assert_info(alphaCell >= 0.0 && alphaCell <= 1.0,
                                         "limited reconstruction alpha must be in [0, 1]");
                        real recWeight = 1.0 - alphaCell;

                        if (recordInc)
                        {
                            if (uRecIsZero)
                                uRecNew[iCell].setZero();
                            else
                                uRecNew[iCell] = uRec[iCell];
                        }
                        else if (settings.SORInstead)
                        {
                            uRec[iCell] *= 1.0 - relax;
                            if (uRecTarget != nullptr)
                                uRec[iCell] += relax * alphaCell * (*uRecTarget)[iCell];
                        }
                        else
                        {
                            uRecNew[iCell] = uRec[iCell] * (1.0 - relax);
                            if (uRecTarget != nullptr)
                                uRecNew[iCell] += relax * alphaCell * (*uRecTarget)[iCell];
                        }

                        auto c2f = mesh->cell2face[iCell];
                        auto matrixAAInvBRow = coefficients_.matrixAAInvB[iCell];
                        auto vectorAInvBRow = coefficients_.vectorAInvB[iCell];
                        for (int ic2f = 0; recWeight > 0 && ic2f < c2f.size(); ic2f++)
                        {
                            index iFace = c2f[ic2f];
                            index iCellOther = mesh->CellFaceOther(iCell, iFace, ic2f);
                            auto faceID = mesh->GetFaceZone(iFace);
                            int if2c = mesh->CellIsFaceBack(iCell, iFace, ic2f) ? 0 : 1;
                            if (iCellOther != UnInitIndex)
                            {
                                Eigen::RowVector<real, nVarsFixed> uOther = u[iCellOther];
                                Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> uRecOther = uRec[iCellOther];
                                this->ApplyPeriodicTransform(if2c, faceID, uOther, uRecOther);
                                if (recordInc)
                                {
                                    if (uRecIsZero)
                                        uRecNew[iCell] -=
                                            (vectorAInvBRow[ic2f] * (uOther - u[iCell].transpose()));
                                    else
                                        uRecNew[iCell] -=
                                            (matrixAAInvBRow[ic2f + 1] * uRecOther +
                                             vectorAInvBRow[ic2f] * (uOther - u[iCell].transpose()));
                                }
                                else if (settings.SORInstead)
                                    uRec[iCell] +=
                                        relax * recWeight *
                                        (matrixAAInvBRow[ic2f + 1] * uRecOther +
                                         vectorAInvBRow[ic2f] * (uOther - u[iCell].transpose()));
                                else
                                    uRecNew[iCell] +=
                                        relax * recWeight *
                                        (matrixAAInvBRow[ic2f + 1] * uRecOther +
                                         vectorAInvBRow[ic2f] * (uOther - u[iCell].transpose()));
                            }
                            else
                            {
                                Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> BCC = GetBoundaryRHS(uRec, u, iCell, iFace, FBoundary);
                                // BCC *= 0;
                                if (recordInc)
                                {
                                    if (uRecIsZero)
                                        ;
                                    else
                                        uRecNew[iCell] -=
                                            matrixAAInvBRow[0] * BCC;
                                }
                                else if (settings.SORInstead && !recordInc)
                                    uRec[iCell] +=
                                        relax * recWeight * matrixAAInvBRow[0] * BCC;
                                else
                                    uRecNew[iCell] +=
                                        relax * recWeight * matrixAAInvBRow[0] * BCC;
                            }
                        }
                        if ((!uRecNew[iCell].allFinite()) || (!uRec[iCell].allFinite()))
                        {
                            DNDS_assert(false);
                        }
                    }

            if (!recordInc)
            {
                if (putIntoNew && settings.SORInstead)
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
                    for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                        uRecNew[iCell].swap(uRec[iCell]);
                if ((!putIntoNew) && (!settings.SORInstead))
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
                    for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                        uRec[iCell].swap(uRecNew[iCell]);
            }
        }

        template <int dim>
        template <int nVarsFixed>
        void VariationalReconstruction<dim>::DoReconstructionIterDiff(
            tURec<nVarsFixed> &uRec,
            tURec<nVarsFixed> &uRecDiff,
            tURec<nVarsFixed> &uRecNew,
            tUDof<nVarsFixed> &u,
            const TFBoundaryDiff<nVarsFixed> &FBoundaryDiff,
            ReconstructionMaskView reconstructionMask)
        {
            auto meshRead = this->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
            const auto meshLegacy = meshRead.LegacyHostView();
            const auto *mesh = &meshLegacy;
            this->ValidateMeshDependencies();
            using namespace Geom;
            using namespace Geom::Elem;
            using namespace Geom::Base;
            int maxNDOF = GetNDof<dim>(settings.maxOrder);
            static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);

            if (settings.maxOrder == 1 && settings.subs2ndOrder != 0)
            {
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    uRecNew[iCell].setZero();
                    if (reconstructionMask.IsMasked(iCell))
                        uRecNew[iCell] = uRecDiff[iCell];
                }
                return;
            }
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                uRecNew[iCell] = uRecDiff[iCell];
                if (reconstructionMask.IsMasked(iCell))
                    continue; // Identity row: derivative of the constraint uRec_i=0.

                auto c2f = mesh->cell2face[iCell];
                auto matrixAAInvBRow = coefficients_.matrixAAInvB[iCell];
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    index iCellOther = mesh->CellFaceOther(iCell, iFace, ic2f);
                    auto faceID = mesh->GetFaceZone(iFace);
                    int if2c = mesh->CellIsFaceBack(iCell, iFace, ic2f) ? 0 : 1;
                    if (iCellOther != UnInitIndex)
                    {
                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> uRecOtherDiff = uRecDiff[iCellOther];
                        this->ApplyPeriodicTransform(if2c, faceID, uRecOtherDiff);
                        uRecNew[iCell] -= matrixAAInvBRow[ic2f + 1] * uRecOtherDiff; // mind the sign
                    }
                    else
                    {
                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> BCC = GetBoundaryRHSDiff(uRec, uRecDiff, u, iCell, iFace, FBoundaryDiff);
                        // BCC *= 0;
                        uRecNew[iCell] -= matrixAAInvBRow[0] * BCC; // mind the sign
                    }
                }
            }
        }

        template <int dim>
        template <int nVarsFixed>
        void VariationalReconstruction<dim>::
            DoReconstructionIterSOR(
                tURec<nVarsFixed> &uRec,
                tURec<nVarsFixed> &uRecInc,
                tURec<nVarsFixed> &uRecNew,
                tUDof<nVarsFixed> &u,
                const TFBoundaryDiff<nVarsFixed> &FBoundaryDiff,
                bool reverse,
                ReconstructionMaskView reconstructionMask)
        {
            auto meshRead = this->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
            const auto meshLegacy = meshRead.LegacyHostView();
            const auto *mesh = &meshLegacy;
            this->ValidateMeshDependencies();
            using namespace Geom;
            using namespace Geom::Elem;
            using namespace Geom::Base;
            int maxNDOF = GetNDof<dim>(settings.maxOrder);
            static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);

            reconstructionMask.Apply(uRecNew);
            for (index iScan = 0; iScan < mesh->NumCell(); iScan++)
            {
                index iCell = iScan;
                if (reverse)
                    iCell = mesh->NumCell() - 1 - iCell;
                if (reconstructionMask.IsMasked(iCell))
                    continue;
                real relax = GetCellAtr(iCell).relax;

                uRecNew[iCell] = (1 - relax) * uRecNew[iCell] + uRecInc[iCell];

                auto c2f = mesh->cell2face[iCell];
                auto matrixAAInvBRow = coefficients_.matrixAAInvB[iCell];
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    index iCellOther = mesh->CellFaceOther(iCell, iFace, ic2f);
                    auto faceID = mesh->GetFaceZone(iFace);
                    int if2c = mesh->CellIsFaceBack(iCell, iFace, ic2f) ? 0 : 1;
                    if (iCellOther != UnInitIndex)
                    {
                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> uRecOtherNew = uRecNew[iCellOther];
                        this->ApplyPeriodicTransform(if2c, faceID, uRecOtherNew);
                        uRecNew[iCell] += relax * matrixAAInvBRow[ic2f + 1] * uRecOtherNew; // mind the sign
                    }
                    else
                    {
                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> BCC = GetBoundaryRHSDiff(uRec, uRecNew, u, iCell, iFace, FBoundaryDiff);
                        // BCC *= 0;
                        uRecNew[iCell] += relax * matrixAAInvBRow[0] * BCC; // mind the sign
                    }
                }
            }
        }
    }
}
