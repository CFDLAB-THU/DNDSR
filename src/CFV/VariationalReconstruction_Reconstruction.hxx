#pragma once

#include "VariationalReconstruction.hpp"

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
            if (settings.intOrderVRIsSame() || settings.functionalSettings.greenGauss1Weight != 0)
                qFace.IntegrationSimple(
                    BCC,
                    [&](auto &vInc, int iG)
                    {
                        RowVectorXR dbv =
                            this->GetIntPointDiffBaseValue(iCell, iFace, -1, iG, std::array<int, 1>{0}, 1);
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
                                this->GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, iG), faceID);
                        Eigen::RowVector<real, nVarsFixed> uIncBV = (uBV - u[iCell]).transpose();
                        if (settings.intOrderVRIsSame())
                            vInc = this->FFaceFunctional(dbv, uIncBV, iFace, iCell, iCell);
                        else
                            vInc.resizeLike(BCC), vInc.setZero();

                        if (settings.functionalSettings.greenGauss1Weight != 0)
                        {
                            // DNDS_assert(false); // not yet implemented
                            vInc += (settings.functionalSettings.greenGauss1Bias * this->GetGreenGauss1WeightOnCell(iCell) *
                                     this->matrixAHalf_GG[iCell].transpose() * this->GetFaceNorm(iFace, iG)(Seq012)) *
                                    uIncBV;
                        }
                        vInc *= this->GetFaceJacobiDet(iFace, iG);
                        // std::cout << faceWeight[iFace].transpose() << std::endl;
                    });
            if (!settings.intOrderVRIsSame())
            {
                auto qFace = Quadrature(mesh->GetFaceElement(iFace), settings.intOrderVRValue());
                tSmallCoords coords;
                mesh->GetCoordsOnFace(iFace, coords);
                qFace.Integration(
                    BCC,
                    [&](auto &vInc, int __xxx_iG, const tPoint &pParam, const Elem::tD01Nj &DiNj) { // todo: cache these for bnd: pPhy JDet norm and dbv
                        BndVRPointCache &bndCacheEntry = bndVRCaches.at(iFace).at(__xxx_iG);
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
                                this->GetFacePointFromCell(iFace, iCell, -1, pPhy), faceID);
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
            if (settings.intOrderVRIsSame() || settings.functionalSettings.greenGauss1Weight != 0)
                qFace.IntegrationSimple(
                    BCC,
                    [&](auto &vInc, int iG)
                    {
                        RowVectorXR dbv =
                            this->GetIntPointDiffBaseValue(iCell, iFace, -1, iG, std::array<int, 1>{0}, 1);
                        Eigen::Vector<real, nVarsFixed> uBL = (dbv * uRec[iCell]).transpose();
                        uBL += u[iCell]; //! need fixing?
                        Eigen::Vector<real, nVarsFixed> uBLDiff = (dbv * uRecDiff[iCell]).transpose();
                        Eigen::Vector<real, nVarsFixed>
                            uBV = FBoundaryDiff(
                                uBL, uBLDiff,
                                u[iCell], iCell, iFace, iG,
                                this->GetFaceNorm(iFace, iG),
                                this->GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, iG), faceID);
                        Eigen::RowVector<real, nVarsFixed> uIncBV = uBV.transpose();
                        if (settings.intOrderVRIsSame())
                            vInc = this->FFaceFunctional(dbv, uIncBV, iFace, iCell, iCell);
                        else
                            vInc.resizeLike(BCC), vInc.setZero();
                        // std::cout << faceWeight[iFace].transpose() << std::endl;
                        if (settings.functionalSettings.greenGauss1Weight != 0)
                        {
                            // DNDS_assert(false); // not yet implemented
                            vInc += (settings.functionalSettings.greenGauss1Bias * this->GetGreenGauss1WeightOnCell(iCell) *
                                     this->matrixAHalf_GG[iCell].transpose() * this->GetFaceNorm(iFace, iG)(Seq012)) *
                                    uIncBV;
                        }
                        vInc *= this->GetFaceJacobiDet(iFace, iG);
                    });
            if (!settings.intOrderVRIsSame())
            {
                auto qFace = Quadrature(mesh->GetFaceElement(iFace), settings.intOrderVRValue());
                tSmallCoords coords;
                mesh->GetCoordsOnFace(iFace, coords);
                qFace.Integration(
                    BCC,
                    [&](auto &vInc, int __xxx_iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                    {
                        BndVRPointCache &bndCacheEntry = bndVRCaches.at(iFace).at(__xxx_iG);
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
                                this->GetFacePointFromCell(iFace, iCell, -1, pPhy), faceID);
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
        void VariationalReconstruction<dim>::DoReconstruction2nd(
            tURec<nVarsFixed> &uRec,
            tUDof<nVarsFixed> &u,
            const TFBoundary<nVarsFixed> &FBoundary,
            int method,
            const std::vector<int> &mask)
        {
            using namespace Geom;
            using namespace Geom::Elem;

            static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>); // note this is gDim!
            static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);

            if (method == 1 || method == 11)
            {
                // simple gauss rule
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
                        index iCellOther = CellFaceOther(iCell, iFace);
                        auto faceID = mesh->GetFaceZone(iFace);
                        int if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
                        auto faceCent = this->GetFaceQuadraturePPhysFromCell(iFace, iCell, if2c, -1);
                        real lThis = (faceCent - this->GetCellBary(iCell)).norm();
                        if (settings.subs2ndOrderGGScheme == 0)
                            lThis = 1;
                        real lThat = lThis;
                        if (iCellOther != UnInitIndex)
                        {
                            lThat = (this->GetOtherCellBaryFromCell(iCell, iCellOther, iFace) - faceCent).norm();
                            if (settings.subs2ndOrderGGScheme == 0)
                                lThat = 1;
                            Eigen::RowVector<real, nVarsFixed> uOther = u[iCellOther].transpose();
                            if (mesh->isPeriodic)
                            {
                                DNDS_assert(FTransPeriodic && FTransPeriodicBack);
                                if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID)) ||
                                    (if2c == 0 && Geom::FaceIDIsPeriodicDonor(faceID))) // I am donor
                                    FTransPeriodic(uOther, faceID);
                                if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) ||
                                    (if2c == 0 && Geom::FaceIDIsPeriodicMain(faceID))) // I am main
                                    FTransPeriodicBack(uOther, faceID);
                            }
                            Eigen::Vector<real, dim> uNorm = this->GetFaceNormFromCell(iFace, iCell, -1, -1)(Seq012) * (CellIsFaceBack(iCell, iFace) ? 1. : -1.);
                            Eigen::Matrix<real, nVarsFixed, dim> gradInc =
                                (uOther.transpose() - u[iCell]) *
                                lThis / (lThis + lThat + verySmallReal) * this->GetFaceArea(iFace) * uNorm.transpose();
                            grad += gradInc;
                            if (method == 11)
                            {
                                mGG(Seq012, dim + ic2f) = -this->GetFaceArea(iFace) / (this->GetCellVol(iCell) + verySmallReal) * uNorm;
                                mGG(dim + ic2f, Seq012) =
                                    (this->GetCellBary(iCell)(Seq012) + this->GetOtherCellBaryFromCell(iCell, iCellOther, iFace)(Seq012) -
                                     2 * faceCent(Seq012))
                                        .transpose();
                                mGG(dim + ic2f, dim + ic2f) = 2;
                                bGG(dim + ic2f, Eigen::all) = uOther + u[iCell].transpose();
                            }
                        }
                        else
                        {
                            auto faceID = mesh->GetFaceZone(iFace);
                            DNDS_assert(FaceIDIsExternalBC(faceID));

                            RowVectorXR dbv =
                                this->GetIntPointDiffBaseValue(
                                    iCell, iFace, -1, -1, std::array<int, 1>{0}, 1);
                            Eigen::Vector<real, nVarsFixed> uBL =
                                (dbv * uRec[iCell]).transpose() * 0.0; // 0.0: uRec should be invalid!!
                            uBL += u[iCell];                           //! need fixing?
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
                                bGG(dim + ic2f, Eigen::all) = (uBV + u[iCell]).transpose();
                            }
                        }
                    }

                    grad /= GetCellVol(iCell);

                    if (method == 11)
                    {
                        // std::cout << mGG << std::endl;
                        // std::cout << bGG << std::endl;
                        // DNDS_assert(false);
                        // auto mGGLU = mGG.colPivHouseholderQr();
                        auto mGGLU = mGG.fullPivLu();
                        DNDS_assert(mGGLU.isInvertible());
                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> xGG = mGGLU.solve(bGG);
                        grad = xGG(Seq012, Eigen::all).transpose();
                    }

                    // tPoint cellBary = GetCellBary(iCell);
                    // Eigen::MatrixXd vvv = (grad * (Geom::RotZ(90) * cellBary)(Eigen::seq(0, dim - 1))).transpose();
                    // std::cout << cellBary.transpose() << " ---- " << vvv << std::endl;
                    // DNDS_assert(vvv(0) < 1e-10);

                    Eigen::Matrix<real, dim, dim> d1bv;
                    d1bv = this->GetIntPointDiffBaseValue(
                        iCell, -1, -1, -1, Seq123, dim + 1)(Eigen::all, Seq012);
                    auto lud1bv = d1bv.partialPivLu();
                    if (lud1bv.rcond() > 1e9)
                        std::cout << "Large Cond " << lud1bv.rcond() << std::endl;
                    if (mask.size() == 0)
                    {
                        uRec[iCell].setZero();
                        uRec[iCell](Seq012, Eigen::all) = lud1bv.solve(grad.transpose());
                    }
                    else
                    {
                        uRec[iCell](Eigen::all, mask).setZero();
                        uRec[iCell](Seq012, mask) = lud1bv.solve(grad.transpose())(Eigen::all, mask);
                    }

                    // std::cout << " g " << std::endl;
                    // std::cout << grad << std::endl;
                    // std::cout << uRec[iCell] << std::endl;
                    // std::cout << d1bv << std::endl;
                    // std::cout << lud1bv.inverse() << std::endl;
                    // std::abort();
                }
            }
            else if (method == 2) //! warning, periodic not implemented here
            {
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
                        index iCellOther = CellFaceOther(iCell, iFace);

                        if (iCellOther != UnInitIndex)
                        {
                            dcs(ic2f, Eigen::all) = (GetOtherCellBaryFromCell(iCell, iCellOther, iFace) - GetCellBary(iCell))(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>))
                                                        .transpose();
                            dus(ic2f, Eigen::all) = (u[iCellOther] - u[iCell]).transpose();
                        }
                        else
                        {
                            auto faceID = mesh->GetFaceZone(iFace);
                            DNDS_assert(FaceIDIsExternalBC(faceID));

                            RowVectorXR dbv =
                                this->GetIntPointDiffBaseValue(
                                    iCell, iFace, -1, -1, std::array<int, 1>{0}, 1);
                            Eigen::Vector<real, nVarsFixed> uBL =
                                (dbv *
                                 uRec[iCell])
                                    .transpose();
                            uBL += u[iCell]; //! need fixing?
                            Eigen::Vector<real, nVarsFixed> uBV =
                                FBoundary(
                                    uBL,
                                    u[iCell], iCell, iFace, -1,
                                    this->GetFaceNorm(iFace, -1),
                                    this->GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, -1), faceID);
                            dus(ic2f, Eigen::all) = (uBV - u[iCell]).transpose();

                            dcs(ic2f, Eigen::all) =
                                ((GetCellBary(iCell) - GetFaceQuadraturePPhys(iFace, -1)).dot(GetFaceNorm(iFace, -1)) * (-2.) * GetFaceNorm(iFace, -1))(
                                    Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>))
                                    .transpose();
                        }
                    }
                    // Eigen::MatrixXd m;
                    // m.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve()
                    auto svd = dcs.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
                    grad = svd.solve(dus).transpose();

                    Eigen::Matrix<real, dim, dim> d1bv;
                    if constexpr (dim == 2)
                        d1bv =
                            this->GetIntPointDiffBaseValue(
                                iCell, -1, -1, -1, std::array<int, 2>{1, 2}, 3);
                    if constexpr (dim == 3)
                        d1bv =
                            this->GetIntPointDiffBaseValue(
                                iCell, -1, -1, -1, std::array<int, 3>{1, 2, 3}, 4);
                    Eigen::Matrix3d m;
                    auto lud1bv = d1bv.partialPivLu();
                    if (lud1bv.rcond() > 1e9)
                        std::cout << "Large Cond " << lud1bv.rcond() << std::endl;
                    uRec[iCell] = lud1bv.solve(grad.transpose());
                }
            }
            else
            {
                DNDS_assert_info(false, "NO SUCH 2nd rec METHOD");
            }
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
            bool uRecIsZero)
        {
            using namespace Geom;
            using namespace Geom::Elem;
            using namespace Geom::Base;
            static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
            int maxNDOF = GetNDof<dim>(settings.maxOrder);
            if (recordInc)
                DNDS_assert_info(putIntoNew, "the -RHS must be put into uRecNew");
            if (settings.maxOrder == 1 && settings.subs2ndOrder != 0)
            {
                if (recordInc)
                    this->DoReconstruction2nd<nVarsFixed>(uRecNew, u, (FBoundary), settings.subs2ndOrder, std::vector<int>());
                else if (putIntoNew)
                    this->DoReconstruction2nd<nVarsFixed>(uRecNew, u, (FBoundary), settings.subs2ndOrder, std::vector<int>());
                else
                    this->DoReconstruction2nd<nVarsFixed>(uRec, u, (FBoundary), settings.subs2ndOrder, std::vector<int>());
                return;
            }
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                real relax = cellAtr[iCell].relax;

                if (recordInc)
                {
                    if (uRecIsZero)
                        uRecNew[iCell].setZero();
                    else
                        uRecNew[iCell] = uRec[iCell];
                }
                else if (settings.SORInstead)
                    uRec[iCell] = uRec[iCell] * ((recordInc ? 0 : 1) - relax);
                else
                    uRecNew[iCell] = uRec[iCell] * ((recordInc ? 0 : 1) - relax);

                auto c2f = mesh->cell2face[iCell];
                auto matrixAAInvBRow = matrixAAInvB[iCell];
                auto vectorAInvBRow = vectorAInvB[iCell];
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    index iCellOther = CellFaceOther(iCell, iFace);
                    auto faceID = mesh->GetFaceZone(iFace);
                    int if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
                    if (iCellOther != UnInitIndex)
                    {
                        Eigen::RowVector<real, nVarsFixed> uOther = u[iCellOther];
                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> uRecOther = uRec[iCellOther];
                        if (mesh->isPeriodic)
                        {
                            DNDS_assert(FTransPeriodic && FTransPeriodicBack);
                            if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID)) ||
                                (if2c == 0 && Geom::FaceIDIsPeriodicDonor(faceID))) // I am donor
                                FTransPeriodic(uOther, faceID), FTransPeriodic(uRecOther, faceID);
                            if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) ||
                                (if2c == 0 && Geom::FaceIDIsPeriodicMain(faceID))) // I am main
                                FTransPeriodicBack(uOther, faceID), FTransPeriodicBack(uRecOther, faceID);
                        }
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
                                relax *
                                (matrixAAInvBRow[ic2f + 1] * uRecOther +
                                 vectorAInvBRow[ic2f] * (uOther - u[iCell].transpose()));
                        else
                            uRecNew[iCell] +=
                                relax *
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
                                relax * matrixAAInvBRow[0] * BCC;
                        else
                            uRecNew[iCell] +=
                                relax * matrixAAInvBRow[0] * BCC;
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
                    for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                        uRecNew[iCell].swap(uRec[iCell]);
                if ((!putIntoNew) && (!settings.SORInstead))
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
            const TFBoundaryDiff<nVarsFixed> &FBoundaryDiff)
        {
            using namespace Geom;
            using namespace Geom::Elem;
            using namespace Geom::Base;
            int maxNDOF = GetNDof<dim>(settings.maxOrder);
            static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);

            if (settings.maxOrder == 1 && settings.subs2ndOrder != 0)
            {
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                    uRecNew[iCell].setZero();
                return;
            }
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                uRecNew[iCell] = uRecDiff[iCell];

                auto c2f = mesh->cell2face[iCell];
                auto matrixAAInvBRow = matrixAAInvB[iCell];
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    index iCellOther = CellFaceOther(iCell, iFace);
                    auto faceID = mesh->GetFaceZone(iFace);
                    int if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
                    if (iCellOther != UnInitIndex)
                    {
                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> uRecOtherDiff = uRecDiff[iCellOther];
                        if (mesh->isPeriodic)
                        {
                            DNDS_assert(FTransPeriodic && FTransPeriodicBack);
                            if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID)) ||
                                (if2c == 0 && Geom::FaceIDIsPeriodicDonor(faceID))) // I am donor
                                FTransPeriodic(uRecOtherDiff, faceID);
                            if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) ||
                                (if2c == 0 && Geom::FaceIDIsPeriodicMain(faceID))) // I am main
                                FTransPeriodicBack(uRecOtherDiff, faceID);
                        }
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
                bool reverse)
        {
            using namespace Geom;
            using namespace Geom::Elem;
            using namespace Geom::Base;
            int maxNDOF = GetNDof<dim>(settings.maxOrder);
            static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);

            for (index iScan = 0; iScan < mesh->NumCell(); iScan++)
            {
                index iCell = iScan;
                if (reverse)
                    iCell = mesh->NumCell() - 1 - iCell;
                real relax = cellAtr[iCell].relax;

                uRecNew[iCell] = (1 - relax) * uRecNew[iCell] + uRecInc[iCell];

                auto c2f = mesh->cell2face[iCell];
                auto matrixAAInvBRow = matrixAAInvB[iCell];
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    index iCellOther = CellFaceOther(iCell, iFace);
                    auto faceID = mesh->GetFaceZone(iFace);
                    int if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
                    if (iCellOther != UnInitIndex)
                    {
                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> uRecOtherNew = uRecNew[iCellOther];
                        if (mesh->isPeriodic)
                        {
                            DNDS_assert(FTransPeriodic && FTransPeriodicBack);
                            if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID)) ||
                                (if2c == 0 && Geom::FaceIDIsPeriodicDonor(faceID))) // I am donor
                                FTransPeriodic(uRecOtherNew, faceID);
                            if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) ||
                                (if2c == 0 && Geom::FaceIDIsPeriodicMain(faceID))) // I am main
                                FTransPeriodicBack(uRecOtherNew, faceID);
                        }
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

#define DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS_EXTERN(dim, nVarsFixed, ext)          \
    namespace DNDS::CFV                                                                         \
    {                                                                                           \
        ext template void VariationalReconstruction<dim>::DoReconstruction2nd<nVarsFixed>(      \
            tURec<nVarsFixed> & uRec,                                                           \
            tUDof<nVarsFixed> &u,                                                               \
            const TFBoundary<nVarsFixed> &FBoundary,                                            \
            int method,                                                                         \
            const std::vector<int> &mask);                                                      \
                                                                                                \
        ext template void VariationalReconstruction<dim>::DoReconstructionIter<nVarsFixed>(     \
            tURec<nVarsFixed> & uRec,                                                           \
            tURec<nVarsFixed> &uRecNew,                                                         \
            tUDof<nVarsFixed> &u,                                                               \
            const TFBoundary<nVarsFixed> &FBoundary,                                            \
            bool putIntoNew,                                                                    \
            bool recordInc,                                                                     \
            bool uRecIsZero);                                                                   \
                                                                                                \
        ext template void VariationalReconstruction<dim>::DoReconstructionIterDiff<nVarsFixed>( \
            tURec<nVarsFixed> & uRec,                                                           \
            tURec<nVarsFixed> &uRecDiff,                                                        \
            tURec<nVarsFixed> &uRecNew,                                                         \
            tUDof<nVarsFixed> &u,                                                               \
            const TFBoundaryDiff<nVarsFixed> &FBoundaryDiff);                                   \
                                                                                                \
        ext template void VariationalReconstruction<dim>::DoReconstructionIterSOR<nVarsFixed>(  \
            tURec<nVarsFixed> & uRec,                                                           \
            tURec<nVarsFixed> &uRecInc,                                                         \
            tURec<nVarsFixed> &uRecNew,                                                         \
            tUDof<nVarsFixed> &u,                                                               \
            const TFBoundaryDiff<nVarsFixed> &FBoundaryDiff,                                    \
            bool reverse);                                                                      \
    }

DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS_EXTERN(2, 4, extern)
DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS_EXTERN(2, 5, extern)
DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS_EXTERN(2, 6, extern)
DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS_EXTERN(2, 7, extern)
DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS_EXTERN(2, Eigen::Dynamic, extern)
DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS_EXTERN(3, 5, extern)
DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS_EXTERN(3, 6, extern)
DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS_EXTERN(3, 7, extern)
DNDS_VARIATIONALRECONSTRUCTION_RECONSTRUCTION_INS_EXTERN(3, Eigen::Dynamic, extern)
