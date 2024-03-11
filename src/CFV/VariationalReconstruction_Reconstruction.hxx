#pragma once

#include "VariationalReconstruction.hpp"

namespace DNDS
{
    namespace CFV
    {

        template <int dim>
        template <int nVarsFixed>
        void VariationalReconstruction<dim>::DoReconstruction2nd(
            tURec<nVarsFixed> &uRec,
            tUDof<nVarsFixed> &u,
            const TFBoundary<nVarsFixed> &FBoundary,
            int method)
        {
            using namespace Geom;
            using namespace Geom::Elem;

            if (method == 1)
            {
                // simple gauss rule
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    int nVars = u[iCell].size();
                    auto c2f = mesh->cell2face[iCell];
                    Eigen::Matrix<real, nVarsFixed, dim> grad;
                    grad.setZero(nVars, dim);

                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                    {
                        index iFace = c2f[ic2f];
                        index iCellOther = CellFaceOther(iCell, iFace);
                        auto faceID = mesh->GetFaceZone(iFace);
                        int if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;

                        if (iCellOther != UnInitIndex)
                        {
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
                            Eigen::Matrix<real, nVarsFixed, dim> gradInc =
                                (uOther.transpose() - u[iCell]) * 0.5 *
                                this->GetFaceNormFromCell(iFace, iCell, -1, -1)(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>)).transpose() *
                                this->GetFaceArea(iFace) * (CellIsFaceBack(iCell, iFace) ? 1. : -1.);

                            grad += gradInc;
                        }
                        else
                        {
                            auto faceID = mesh->GetFaceZone(iFace);
                            DNDS_assert(FaceIDIsExternalBC(faceID));

                            Eigen::Matrix<real, 1, Eigen::Dynamic> dbv =
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
                                    u[iCell], iCell, iFace,
                                    this->GetFaceNorm(iFace, -1),
                                    this->GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, -1), faceID);
                            grad += (uBV - u[iCell]) * 0.5 *
                                    this->GetFaceNorm(iFace, -1)(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>)).transpose() *
                                    this->GetFaceArea(iFace);
                        }
                    }

                    grad /= GetCellVol(iCell);
                    // tPoint cellBary = GetCellBary(iCell);
                    // Eigen::MatrixXd vvv = (grad * (Geom::RotZ(90) * cellBary)(Eigen::seq(0, dim - 1))).transpose();
                    // std::cout << cellBary.transpose() << " ---- " << vvv << std::endl;
                    // DNDS_assert(vvv(0) < 1e-10);

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

                            Eigen::Matrix<real, 1, Eigen::Dynamic> dbv =
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
                                    u[iCell], iCell, iFace,
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
            bool recordInc)
        {
            using namespace Geom;
            using namespace Geom::Elem;
            static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
            int maxNDOF = GetNDof<dim>(settings.maxOrder);
            if (recordInc)
                DNDS_assert_info(putIntoNew, "the -RHS must be put into uRecNew");
            if (settings.maxOrder == 1 && settings.subs2ndOrder != 0)
            {
                if (recordInc)
                    this->DoReconstruction2nd<nVarsFixed>(uRecNew, u, (FBoundary), settings.subs2ndOrder);
                else if (putIntoNew)
                    this->DoReconstruction2nd<nVarsFixed>(uRecNew, u, (FBoundary), settings.subs2ndOrder);
                else
                    this->DoReconstruction2nd<nVarsFixed>(uRec, u, (FBoundary), settings.subs2ndOrder);
                return;
            }
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                real relax = cellAtr[iCell].relax;

                if (recordInc)
                    uRecNew[iCell] = uRec[iCell];
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
                            uRecNew[iCell] -=
                                (matrixAAInvBRow[ic2f + 1] * uRecOther +
                                 vectorAInvBRow[ic2f] * (uOther - u[iCell].transpose()));
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
                        auto faceID = mesh->GetFaceZone(iFace);
                        DNDS_assert(FaceIDIsExternalBC(faceID));

                        int nVars = u[iCell].size();

                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> BCC;
                        BCC.setZero(uRec[iCell].rows(), uRec[iCell].cols());

                        auto qFace = this->GetFaceQuad(iFace);
                        qFace.IntegrationSimple(
                            BCC,
                            [&](auto &vInc, int iG)
                            {
                                Eigen::Matrix<real, 1, Eigen::Dynamic> dbv =
                                    this->GetIntPointDiffBaseValue(
                                        iCell, iFace, -1, iG, std::array<int, 1>{0}, 1);
                                Eigen::Vector<real, nVarsFixed> uBL =
                                    (dbv *
                                     uRec[iCell])
                                        .transpose();
                                uBL += u[iCell]; //! need fixing?
                                Eigen::Vector<real, nVarsFixed> uBV =
                                    FBoundary(
                                        uBL,
                                        u[iCell], iCell, iFace,
                                        this->GetFaceNorm(iFace, iG),
                                        this->GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, iG), faceID);
                                Eigen::RowVector<real, nVarsFixed> uIncBV = (uBV - u[iCell]).transpose();
                                vInc = this->FFaceFunctional(dbv, uIncBV, iFace, iG, iCell, iCell);
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
                        // BCC *= 0;
                        if (recordInc)
                            uRecNew[iCell] -=
                                matrixAAInvBRow[0] * BCC;
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
                        auto faceID = mesh->GetFaceZone(iFace);
                        DNDS_assert(FaceIDIsExternalBC(faceID));

                        int nVars = (int)u[iCell].size();

                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> BCC;
                        BCC.setZero(uRec[iCell].rows(), uRec[iCell].cols());

                        auto qFace = this->GetFaceQuad(iFace);
                        qFace.IntegrationSimple(
                            BCC,
                            [&](auto &vInc, int iG)
                            {
                                Eigen::Matrix<real, 1, Eigen::Dynamic> dbv =
                                    this->GetIntPointDiffBaseValue(
                                        iCell, iFace, -1, iG, std::array<int, 1>{0}, 1);
                                Eigen::Vector<real, nVarsFixed> uBL =
                                    (dbv *
                                     uRec[iCell])
                                        .transpose();
                                uBL += u[iCell]; //! need fixing?
                                Eigen::Vector<real, nVarsFixed> uBLDiff =
                                    (dbv *
                                     uRecDiff[iCell])
                                        .transpose();
                                Eigen::Vector<real, nVarsFixed>
                                    uBV =
                                        FBoundaryDiff(
                                            uBL,
                                            uBLDiff,
                                            u[iCell], iCell, iFace,
                                            this->GetFaceNorm(iFace, iG),
                                            this->GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, iG), faceID);
                                Eigen::RowVector<real, nVarsFixed> uIncBV = uBV.transpose();
                                vInc = this->FFaceFunctional(dbv, uIncBV, iFace, iG, iCell, iCell) * this->GetFaceJacobiDet(iFace, iG);
                                // std::cout << faceWeight[iFace].transpose() << std::endl;
                                if (settings.functionalSettings.greenGauss1Weight != 0)
                                {
                                    // DNDS_assert(false); // not yet implemented
                                    vInc += (settings.functionalSettings.greenGauss1Bias * this->GetGreenGauss1WeightOnCell(iCell) *
                                             this->matrixAHalf_GG[iCell].transpose() * this->GetFaceNorm(iFace, iG)(Seq012)) *
                                            uIncBV;
                                }
                            });
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
                        auto faceID = mesh->GetFaceZone(iFace);
                        DNDS_assert(FaceIDIsExternalBC(faceID));

                        int nVars = u[iCell].size();

                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> BCC;
                        BCC.setZero(uRec[iCell].rows(), uRec[iCell].cols());

                        auto qFace = this->GetFaceQuad(iFace);
                        qFace.IntegrationSimple(
                            BCC,
                            [&](auto &vInc, int iG)
                            {
                                Eigen::Matrix<real, 1, Eigen::Dynamic> dbv =
                                    this->GetIntPointDiffBaseValue(
                                        iCell, iFace, -1, iG, std::array<int, 1>{0}, 1);
                                Eigen::Vector<real, nVarsFixed> uBL =
                                    (dbv *
                                     uRec[iCell])
                                        .transpose();
                                uBL += u[iCell]; //! need fixing?
                                Eigen::Vector<real, nVarsFixed> uBLDiff =
                                    (dbv *
                                     uRecNew[iCell])
                                        .transpose();
                                Eigen::Vector<real, nVarsFixed>
                                    uBV =
                                        FBoundaryDiff(
                                            uBL,
                                            uBLDiff,
                                            u[iCell], iCell, iFace,
                                            this->GetFaceNorm(iFace, iG),
                                            this->GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, iG), faceID);
                                Eigen::RowVector<real, nVarsFixed> uIncBV = uBV.transpose();
                                vInc = this->FFaceFunctional(dbv, uIncBV, iFace, iG, iCell, iCell) * this->GetFaceJacobiDet(iFace, iG);
                                // std::cout << faceWeight[iFace].transpose() << std::endl;
                                if (settings.functionalSettings.greenGauss1Weight != 0)
                                {
                                    // DNDS_assert(false); // not yet implemented
                                    vInc += (settings.functionalSettings.greenGauss1Bias * this->GetGreenGauss1WeightOnCell(iCell) *
                                             this->matrixAHalf_GG[iCell].transpose() * this->GetFaceNorm(iFace, iG)(Seq012)) *
                                            uIncBV;
                                }
                            });
                        // BCC *= 0;
                        uRecNew[iCell] += relax * matrixAAInvBRow[0] * BCC; // mind the sign
                    }
                }
            }
        }
    }
}
