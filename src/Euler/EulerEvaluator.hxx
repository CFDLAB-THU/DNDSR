#pragma once

#include "DNDS/Defines.hpp" // for correct  DNDS_SWITCH_INTELLISENSE
#include "EulerEvaluator.hpp"
#include "DNDS/HardEigen.hpp"
#include "SpecialFields.hpp"

namespace DNDS::Euler
{

    static const auto model = NS_SA; // to be hidden by template params

    template <EulerModel model>
    void EulerEvaluator<model>::LUSGSMatrixInit(
        ArrayDOFV<nVarsFixed> &JDiag,
        ArrayDOFV<nVarsFixed> &JSource,
        ArrayDOFV<1> &dTau, real dt, real alphaDiag,
        ArrayDOFV<nVarsFixed> &u,
        ArrayRECV<nVarsFixed> &uRec,
        int jacobianCode,
        real t)
    {
        // TODO: for code0: flux jacobian with lambdaFace, and source jacobian with integration, only diagpart dealt with
        DNDS_assert(jacobianCode == 0);
        int cnvars = nVars;
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            auto c2f = mesh->cell2face[iCell];

            // LUSGS diag part
            real fpDivisor = 1.0 / dTau[iCell](0) + 1.0 / dt;
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                fpDivisor += (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) * lambdaFace[iFace] / vfv->GetCellVol(iCell);
            }
            JDiag[iCell].setConstant(fpDivisor);

            // std::cout << fpDivisor << std::endl;

            // jacobian diag

            if (!settings.ignoreSourceTerm)
                JDiag[iCell] += alphaDiag * JSource[iCell];

            // jacobianCellInv[iCell] = jacobianCell[iCell].partialPivLu().inverse();

            // std::cout << "jacobian Diag\n"
            //           << jacobianCell[iCell] << std::endl;
            // std::cout << dTau[iCell] << "\n";
        }
        // exit(-1);
    }

    template <EulerModel model>
    void EulerEvaluator<model>::LUSGSMatrixVec(
        real alphaDiag,
        ArrayDOFV<nVarsFixed> &u,
        ArrayDOFV<nVarsFixed> &uInc,
        ArrayDOFV<nVarsFixed> &JDiag,
        ArrayDOFV<nVarsFixed> &AuInc)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "LUSGSMatrixVec 1");
        int cnvars = nVars;
        for (index iScan = 0; iScan < mesh->NumCell(); iScan++)
        {
            index iCell = iScan;
            // iCell = (*vfv->SOR_iScan2iCell)[iCell];//TODO: add rb-sor

            auto c2f = mesh->cell2face[iCell];
            TU uIncNewBuf(cnvars);
            uIncNewBuf.setZero(); // norhs
            auto uINCi = uInc[iCell];

            if (uINCi.hasNaN())
            {
                std::cout << uINCi << std::endl;
                DNDS_assert(false);
            }

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto f2c = mesh->face2cell[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                if (iCellOther != UnInitIndex)
                {

                    if (true)
                    {
                        TU uINCj = uInc[iCellOther];
                        TU uj = u[iCellOther];
                        this->UFromOtherCell(uINCj, iFace, iCell, iCellOther, iCellAtFace);
                        this->UFromOtherCell(uj, iFace, iCell, iCellOther, iCellAtFace);
                        TU fInc;
                        {
                            TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCellOther, iCellAtFace, -1)(Seq012) *
                                            (iCellAtFace ? -1 : 1); // faces out

                            // fInc = fluxJacobian0_Right(
                            //            u[iCellOther],
                            //            unitNorm,
                            //            BoundaryType::Inner) *
                            //        uInc[iCellOther]; //! always inner here
                            fInc = fluxJacobian0_Right_Times_du(
                                uj,
                                unitNorm, GetFaceVGrid(iFace, -1),
                                Geom::BC_ID_INTERNAL, uINCj, lambdaFace[iFace], lambdaFaceC[iFace]); //! always inner here
                        }

                        uIncNewBuf -= (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) *
                                      (fInc);
                        if (uIncNewBuf.hasNaN() || (!uIncNewBuf.allFinite()))
                        {
                            std::cout
                                << fInc.transpose() << std::endl
                                << uInc[iCellOther].transpose() << std::endl;
                            DNDS_assert(!(uIncNewBuf.hasNaN() || (!uIncNewBuf.allFinite())));
                        }
                    }
                }
            }
            // uIncNewBuf /= fpDivisor;
            // uIncNew[iCell] = uIncNewBuf;
            auto AuIncI = AuInc[iCell];
            AuIncI = JDiag[iCell].array() * uInc[iCell].array() - uIncNewBuf.array();

            if (AuIncI.hasNaN())
            {
                std::cout << AuIncI.transpose() << std::endl
                          << uINCi.transpose() << std::endl
                          << u[iCell].transpose() << std::endl
                          << JDiag[iCell] << std::endl
                          << iCell << std::endl;
                DNDS_assert(!AuInc[iCell].hasNaN());
            }
        }
        DNDS_MPI_InsertCheck(u.father->getMPI(), "LUSGSMatrixVec -1");
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>, )
    void EulerEvaluator<model>::LUSGSMatrixToJacobianLU(
        real alphaDiag,
        ArrayDOFV<nVarsFixed> &u,
        ArrayDOFV<nVarsFixed> &JDiag,
        JacobianLocalLU<nVarsFixed> &jacLU)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "LUSGSMatrixToJacobianLU 1");
        int cnvars = nVars;
        jacLU.setZero();
        for (index iScan = 0; iScan < mesh->NumCell(); iScan++)
        {
            index iCell = iScan;
            // iCell = (*vfv->SOR_iScan2iCell)[iCell];//TODO: add rb-sor
            auto c2f = mesh->cell2face[iCell];
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto f2c = mesh->face2cell[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                rowsize iCellAtFace = f2c[0] == iCell ? 0 : 1;
                if (iCellOther != UnInitIndex && iCellOther != iCell && iCellOther < mesh->NumCell())
                {
                    TU uj = u[iCellOther];
                    this->UFromOtherCell(uj, iFace, iCell, iCellOther, iCellAtFace);
                    int iC2CInLocal = -1;
                    for (int ic2c = 0; ic2c < mesh->cell2cellFaceVLocal[iCell].size(); ic2c++)
                        if (iCellOther == mesh->cell2cellFaceVLocal[iCell][ic2c])
                            iC2CInLocal = ic2c; // TODO: pre-search this
                    DNDS_assert(iC2CInLocal != -1);
                    TJacobianU jacIJ;
                    {
                        TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCellOther, iCellAtFace, -1)(Seq012) *
                                        (iCellAtFace ? -1 : 1);        // faces out
                        jacIJ = fluxJacobian0_Right_Times_du_AsMatrix( // unitnorm and uj are both respect with this cell
                            uj,
                            unitNorm, GetFaceVGrid(iFace, -1),
                            Geom::BC_ID_INTERNAL, lambdaFace[iFace], lambdaFaceC[iFace]); //! always inner here
                    }
                    auto faceID = mesh->GetFaceZone(iFace);
                    mesh->CellOtherCellPeriodicHandle(
                        iFace, iCellAtFace,
                        [&]()
                        { jacIJ(Eigen::all, Seq123) =
                              mesh->periodicInfo.TransVectorBack<dim, nVarsFixed>(
                                                    jacIJ(Eigen::all, Seq123).transpose(), faceID)
                                  .transpose(); },
                        [&]()
                        { jacIJ(Eigen::all, Seq123) =
                              mesh->periodicInfo.TransVector<dim, nVarsFixed>(
                                                    jacIJ(Eigen::all, Seq123).transpose(), faceID)
                                  .transpose(); });
                    jacLU.LDU(iCell, symLU->cell2cellFaceVLocal2FullRowPos[iCell][iC2CInLocal]) =
                        (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) * jacIJ;
                }
            }
            jacLU.GetDiag(iCell) = JDiag[iCell].asDiagonal();
        }
        jacLU.InPlaceDecompose();
        DNDS_MPI_InsertCheck(u.father->getMPI(), "LUSGSMatrixToJacobianLU -1");
    }

    template <EulerModel model>
    void EulerEvaluator<model>::UpdateLUSGSForward(
        real alphaDiag,
        ArrayDOFV<nVarsFixed> &rhs,
        ArrayDOFV<nVarsFixed> &u,
        ArrayDOFV<nVarsFixed> &uInc,
        ArrayDOFV<nVarsFixed> &JDiag,
        ArrayDOFV<nVarsFixed> &uIncNew)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateLUSGSForward 1");
        int cnvars = nVars;
        index nCellDist = mesh->NumCell();
        for (index iScan = 0; iScan < nCellDist; iScan++)
        {
            index iCell = iScan;
            iCell = iScan; // TODO: add rb-sor

            auto c2f = mesh->cell2face[iCell];
            TU uIncNewBuf(nVars);
            auto RHSI = rhs[iCell];
            // std::cout << rhs[iCell](0) << std::endl;
            uIncNewBuf = RHSI;

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto f2c = mesh->face2cell[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                if (iCellOther != UnInitIndex)
                {

                    index iScanOther = iCellOther; // TODO: add rb-sor
                    if (iScanOther < iScan)
                    {
                        TU fInc;
                        TU uINCj = uInc[iCellOther];
                        TU uj = u[iCellOther];
                        this->UFromOtherCell(uINCj, iFace, iCell, iCellOther, iCellAtFace);
                        this->UFromOtherCell(uj, iFace, iCell, iCellOther, iCellAtFace);

                        {
                            TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCellOther, iCellAtFace, -1)(Seq012) *
                                            (iCellAtFace ? -1 : 1); // faces out

                            fInc = fluxJacobian0_Right_Times_du(
                                uj,
                                unitNorm, GetFaceVGrid(iFace, -1),
                                Geom::BC_ID_INTERNAL, uINCj, lambdaFace[iFace], lambdaFaceC[iFace]); //! always inner here
                        }

                        uIncNewBuf -= (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) *
                                      (fInc);

                        if ((!uIncNewBuf.allFinite()))
                        {
                            std::cout << RHSI.transpose() << std::endl
                                      << fInc.transpose() << std::endl
                                      << uINCj.transpose() << std::endl;
                            DNDS_assert(false);
                        }
                    }
                }
            }
            auto uIncNewI = uIncNew[iCell];
            uIncNewI.array() = JDiag[iCell].array().inverse() * uIncNewBuf.array();

            if (uIncNewI.hasNaN())
            {
                std::cout << uIncNewI.transpose() << std::endl
                          << uIncNewBuf.transpose() << std::endl
                          << JDiag[iCell] << std::endl
                          << iCell << std::endl;
                DNDS_assert(!uIncNew[iCell].hasNaN());
            }
        }
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateLUSGSForward -1");
        // exit(-1);
    }

    template <EulerModel model>
    void EulerEvaluator<model>::UpdateLUSGSBackward(
        real alphaDiag,
        ArrayDOFV<nVarsFixed> &rhs,
        ArrayDOFV<nVarsFixed> &u,
        ArrayDOFV<nVarsFixed> &uInc,
        ArrayDOFV<nVarsFixed> &JDiag,
        ArrayDOFV<nVarsFixed> &uIncNew)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateLUSGSBackward 1");
        int cnvars = nVars;
        index nCellDist = mesh->NumCell();
        for (index iScan = nCellDist - 1; iScan >= 0; iScan--)
        {
            index iCell = iScan;
            iCell = iScan;

            auto c2f = mesh->cell2face[iCell];
            TU uIncNewBuf(cnvars);
            uIncNewBuf.setZero(); // backward
            auto RHSI = rhs[iCell];

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto f2c = mesh->face2cell[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                if (iCellOther != UnInitIndex)
                {
                    index iScanOther = iCellOther;
                    if (iScanOther > iScan) // backward
                    {
                        TU fInc;
                        TU uINCj = uInc[iCellOther];
                        TU uj = u[iCellOther];
                        this->UFromOtherCell(uINCj, iFace, iCell, iCellOther, iCellAtFace);
                        this->UFromOtherCell(uj, iFace, iCell, iCellOther, iCellAtFace);

                        {
                            TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCellOther, iCellAtFace, -1)(Seq012) *
                                            (iCellAtFace ? -1 : 1); // faces out

                            fInc = fluxJacobian0_Right_Times_du(
                                uj,
                                unitNorm, GetFaceVGrid(iFace, -1),
                                Geom::BC_ID_INTERNAL, uINCj, lambdaFace[iFace], lambdaFaceC[iFace]); //! always inner here
                        }

                        uIncNewBuf -= (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) *
                                      (fInc);
                    }
                }
            }
            auto uIncNewI = uIncNew[iCell];
            uIncNewI.array() += JDiag[iCell].array().inverse() * uIncNewBuf.array();
        }
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateLUSGSBackward -1");
    }

    template <EulerModel model>
    void EulerEvaluator<model>::UpdateSGS(
        real alphaDiag,
        ArrayDOFV<nVarsFixed> &rhs,
        ArrayDOFV<nVarsFixed> &u,
        ArrayDOFV<nVarsFixed> &uInc,
        ArrayDOFV<nVarsFixed> &uIncNew,
        ArrayDOFV<nVarsFixed> &JDiag,
        bool forward, TU &sumInc)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateSGS 1");
        int cnvars = nVars;
        index nCellDist = mesh->NumCell();
        sumInc.setZero(cnvars);
        for (index iScan = 0; iScan < nCellDist; iScan++)
        {
            index iCell = iScan;
            iCell = forward ? iScan : nCellDist - 1 - iScan; // TODO: add rb-sor

            auto c2f = mesh->cell2face[iCell];
            TU uIncNewBuf(nVars);
            auto RHSI = rhs[iCell];
            // std::cout << rhs[iCell](0) << std::endl;
            uIncNewBuf = RHSI;

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto f2c = mesh->face2cell[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                if (iCellOther != UnInitIndex)
                {
                    index iScanOther = forward ? iCellOther : nCellDist - 1 - iCellOther; // TODO: add rb-sor
                    if (iCell != iCellOther)
                    {
                        TU fInc;
                        TU uINCj = uInc[iCellOther];
                        TU uj = u[iCellOther];
                        this->UFromOtherCell(uINCj, iFace, iCell, iCellOther, iCellAtFace);
                        this->UFromOtherCell(uj, iFace, iCell, iCellOther, iCellAtFace);

                        {
                            TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCellOther, iCellAtFace, -1)(Seq012) *
                                            (iCellAtFace ? -1 : 1); // faces out

                            fInc = fluxJacobian0_Right_Times_du(
                                uj,
                                unitNorm, GetFaceVGrid(iFace, -1),
                                Geom::BC_ID_INTERNAL, uINCj, lambdaFace[iFace], lambdaFaceC[iFace]); //! always inner here
                        }

                        uIncNewBuf -= (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) *
                                      (fInc);

                        if ((!uIncNewBuf.allFinite()))
                        {
                            std::cout << RHSI.transpose() << std::endl
                                      << fInc.transpose() << std::endl
                                      << uINCj.transpose() << std::endl;
                            DNDS_assert(false);
                        }
                    }
                }
            }
            auto uIncNewI = uIncNew[iCell];
            TU uIncOld = uIncNewI;

            uIncNewI.array() = JDiag[iCell].array().inverse() * uIncNewBuf.array();
            sumInc.array() += (uIncNewI - uIncOld).array().abs();

            if (uIncNewI.hasNaN())
            {
                std::cout << uIncNewI.transpose() << std::endl
                          << uIncNewBuf.transpose() << std::endl
                          << JDiag[iCell] << std::endl
                          << iCell << std::endl;
                DNDS_assert(!uInc[iCell].hasNaN());
            }
            // if (iScan == 100)
        }
        TU sumIncAll(cnvars);
        // std::abort();
        MPI::Allreduce(sumInc.data(), sumIncAll.data(), sumInc.size(), DNDS_MPI_REAL, MPI_SUM, rhs.father->getMPI().comm);
        sumInc = sumIncAll;
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateSGS -1");
        // exit(-1);
    }

    template <EulerModel model>
    void EulerEvaluator<model>::UpdateSGSWithRec(
        real alphaDiag,
        ArrayDOFV<nVarsFixed> &rhs,
        ArrayDOFV<nVarsFixed> &u,
        ArrayRECV<nVarsFixed> &uRec,
        ArrayDOFV<nVarsFixed> &uInc,
        ArrayRECV<nVarsFixed> &uRecInc,
        ArrayDOFV<nVarsFixed> &JDiag,
        bool forward, TU &sumInc)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateSGS 1");
        int cnvars = nVars;
        index nCellDist = mesh->NumCell();
        sumInc.setZero(cnvars);
        for (index iScan = 0; iScan < nCellDist; iScan++)
        {
            index iCell = iScan;
            iCell = forward ? iScan : nCellDist - 1 - iScan; // TODO: add rb-sor

            auto c2f = mesh->cell2face[iCell];
            TU uIncNewBuf(nVars);
            auto RHSI = rhs[iCell];
            // std::cout << rhs[iCell](0) << std::endl;
            uIncNewBuf = RHSI;

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto f2c = mesh->face2cell[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                if (iCellOther != UnInitIndex)
                {
                    index iScanOther = forward ? iCellOther : nCellDist - 1 - iCellOther; // TODO: add rb-sor
                    if (iCell != iCellOther)
                    {
                        TU fInc, fIncS;
                        auto uINCj = uInc[iCellOther];
                        TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCellOther, iCellAtFace, -1)(Seq012) *
                                        (iCellAtFace ? -1 : 1); // faces out
                        {
                            fInc = fluxJacobian0_Right_Times_du(
                                u[iCellOther], //! TODO periodic here
                                unitNorm, GetFaceVGrid(iFace, -1),
                                Geom::BC_ID_INTERNAL, uINCj, lambdaFace[iFace], lambdaFaceC[iFace]); //! always inner here
                        }
                        {
                            TU uRecSLInc =
                                (vfv->GetIntPointDiffBaseValue(iCell, iFace, iCellAtFace, -1, std::array<int, 1>{0}, 1) *
                                 uRecInc[iCell])
                                    .transpose();
                            TU uRecSRInc =
                                (vfv->GetIntPointDiffBaseValue(iCellOther, iFace, 1 - iCellAtFace, -1, std::array<int, 1>{0}, 1) *
                                 uRecInc[iCellOther]) //! TODO periodic here
                                    .transpose();
                            TU fIncSL = fluxJacobianC_Right_Times_du(u[iCell], unitNorm, GetFaceVGrid(iFace, -1), Geom::BC_ID_INTERNAL, uRecSLInc);
                            TU fIncSR = fluxJacobianC_Right_Times_du(u[iCellOther], unitNorm, GetFaceVGrid(iFace, -1), Geom::BC_ID_INTERNAL, uRecSRInc);
                            fIncS = fIncSL + fIncSR + lambdaFaceC[iFace] * (uRecSLInc - uRecSRInc);
                        }

                        uIncNewBuf -= (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) *
                                      (fInc);
                        uIncNewBuf -= (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) *
                                      (fIncS);

                        if ((!uIncNewBuf.allFinite()))
                        {
                            std::cout << RHSI.transpose() << std::endl
                                      << fInc.transpose() << std::endl
                                      << uINCj.transpose() << std::endl;
                            DNDS_assert(false);
                        }
                    }
                }
            }
            auto uIncNewI = uInc[iCell];
            TU uIncOld = uIncNewI;

            uIncNewI.array() = JDiag[iCell].array().inverse() * uIncNewBuf.array();
            sumInc.array() += (uIncNewI - uIncOld).array().abs();

            if (uIncNewI.hasNaN())
            {
                std::cout << uIncNewI.transpose() << std::endl
                          << uIncNewBuf.transpose() << std::endl
                          << JDiag[iCell] << std::endl
                          << iCell << std::endl;
                DNDS_assert(!uInc[iCell].hasNaN());
            }
        }
        TU sumIncAll(cnvars);
        MPI::Allreduce(sumInc.data(), sumIncAll.data(), sumInc.size(), DNDS_MPI_REAL, MPI_SUM, rhs.father->getMPI().comm);
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateSGS -1");
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>, template <>
    )
    void EulerEvaluator<model>::LUSGSMatrixSolveJacobianLU(
        real alphaDiag,
        ArrayDOFV<nVarsFixed> &rhs,
        ArrayDOFV<nVarsFixed> &u,
        ArrayDOFV<nVarsFixed> &uInc,
        ArrayDOFV<nVarsFixed> &uIncNew,
        ArrayDOFV<nVarsFixed> &bBuf,
        ArrayDOFV<nVarsFixed> &JDiag,
        JacobianLocalLU<nVarsFixed> &jacLU,
        TU &sumInc)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "LUSGSMatrixSolveJacobianLU 1");
        int cnvars = nVars;
        index nCellDist = mesh->NumCell();
        sumInc.setZero(cnvars);
        for (index iScan = 0; iScan < nCellDist; iScan++) // update the ghost part (non proc-block) rhs
        {
            index iCell = iScan;
            auto c2f = mesh->cell2face[iCell];
            bBuf[iCell] = rhs[iCell];
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto f2c = mesh->face2cell[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                if (iCellOther != UnInitIndex && iCell != iCellOther
                    // if is a ghost neighbour
                    && iCellOther >= mesh->NumCell())
                {
                    TU fInc;
                    TU uINCj = uInc[iCellOther];
                    TU uj = u[iCellOther];
                    this->UFromOtherCell(uINCj, iFace, iCell, iCellOther, iCellAtFace);
                    this->UFromOtherCell(uj, iFace, iCell, iCellOther, iCellAtFace);
                    {
                        TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCellOther, iCellAtFace, -1)(Seq012) *
                                        (iCellAtFace ? -1 : 1); // faces out

                        fInc = fluxJacobian0_Right_Times_du(
                            uj,
                            unitNorm, GetFaceVGrid(iFace, -1),
                            Geom::BC_ID_INTERNAL, uINCj, lambdaFace[iFace], lambdaFaceC[iFace]); //! always inner here
                    }

                    bBuf[iCell] -= (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) *
                                   (fInc);
                }
            }
            // TU uIncOld = uIncNew[iCell];
            // uIncNew[iCell] = JDiag[iCell].array().inverse() * bBuf[iCell].array();
            // sumInc.array() += (uIncNew[iCell] - uIncOld).array().abs();
        }
        jacLU.Solve(bBuf, uIncNew); // top-diagonal solve

        DNDS_assert(uIncNew.father.get() != uInc.father.get()); // no aliasing
        uInc -= uIncNew;
        sumInc = uInc.componentWiseNorm1();
        // TU sumIncAll(cnvars);
        // MPI::Allreduce(sumInc.data(), sumIncAll.data(), sumInc.size(), DNDS_MPI_REAL, MPI_SUM, rhs.father->getMPI().comm);
        // sumInc = sumIncAll;

        DNDS_MPI_InsertCheck(u.father->getMPI(), "LUSGSMatrixSolveJacobianLU -1");
        // exit(-1);
    }

    template <EulerModel model>
    void EulerEvaluator<model>::InitializeUDOF(ArrayDOFV<nVarsFixed> &u)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        Eigen::VectorXd initConstVal = this->settings.farFieldStaticValue;
        u.setConstant(initConstVal);
        if (model == EulerModel::NS_SA || model == NS_SA_3D)
        {
            for (int iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                auto c2f = mesh->cell2face[iCell];
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    if (pBCHandler->GetTypeFromID(mesh->GetFaceZone(iFace)) == EulerBCType::BCWall)
                        u[iCell](I4 + 1) *= 1.0; // ! not fixing first layer!
                }
            }
        }

        switch (settings.specialBuiltinInitializer)
        {
        case 1: // for RT problem
            DNDS_assert(model == NS || model == NS_2D || model == NS_3D);
            if constexpr (model == NS || model == NS_2D)
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    Geom::tPoint pos = vfv->GetCellBary(iCell);
                    real gamma = settings.idealGasProperty.gamma;
                    real rho = 2;
                    real p = 1 + 2 * pos(1);
                    if (pos(1) >= 0.5)
                    {
                        rho = 1;
                        p = 1.5 + pos(1);
                    }
                    real v = -0.025 * sqrt(gamma * p / rho) * std::cos(8 * pi * pos(0));
                    if constexpr (dim == 3)
                        u[iCell] = Eigen::Vector<real, 5>{rho, 0, rho * v, 0, 0.5 * rho * sqr(v) + p / (gamma - 1)};
                    else
                        u[iCell] = Eigen::Vector<real, 4>{rho, 0, rho * v, 0.5 * rho * sqr(v) + p / (gamma - 1)};
                }
            else if constexpr (model == NS_3D)
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    Geom::tPoint pos = vfv->GetCellBary(iCell);
                    real gamma = settings.idealGasProperty.gamma;
                    real rho = 2;
                    real p = 1 + 2 * pos(1);
                    if (pos(1) >= 0.5)
                    {
                        rho = 1;
                        p = 1.5 + pos(1);
                    }
                    real v = -0.025 * sqrt(gamma * p / rho) * std::cos(8 * pi * pos(0)) * std::cos(8 * pi * pos(2));
                    u[iCell] = Eigen::Vector<real, 5>{rho, 0, rho * v, 0, 0.5 * rho * sqr(v) + p / (gamma - 1)};
                }
            break;
        case 2: // for IV10 problem
        case 202:
            DNDS_assert(model == NS || model == NS_2D);
            if constexpr (model == NS || model == NS_2D)
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    Geom::tPoint pos = vfv->GetCellBary(iCell);
                    auto c2n = mesh->cell2node[iCell];
                    auto gCell = vfv->GetCellQuad(iCell);
                    TU um;
                    um.resizeLike(u[iCell]);
                    um.setZero();
                    gCell.IntegrationSimple(
                        um,
                        [&](TU &inc, int ig)
                        {
                            Geom::tPoint pPhysics = vfv->GetCellQuadraturePPhys(iCell, ig);
                            if (settings.specialBuiltinInitializer == 2)
                                inc = SpecialFields::IsentropicVortex10(*this, pPhysics, 0, nVars);
                            else if (settings.specialBuiltinInitializer == 202)
                                inc = SpecialFields::IsentropicVortex30(*this, pPhysics, 0, nVars);
                            else
                                DNDS_assert(false);
                            inc *= vfv->GetCellJacobiDet(iCell, ig); // don't forget this
                        });
                    u[iCell] = um / vfv->GetCellVol(iCell); // mean value
                }
            break;
        case 201: // for IVCent problem
            DNDS_assert(model == NS || model == NS_2D);
            if constexpr (model == NS || model == NS_2D)
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    Geom::tPoint pos = vfv->GetCellBary(iCell);
                    auto c2n = mesh->cell2node[iCell];
                    auto gCell = vfv->GetCellQuad(iCell);
                    TU um;
                    um.resizeLike(u[iCell]);
                    um.setZero();
                    gCell.IntegrationSimple(
                        um,
                        [&](TU &inc, int ig)
                        {
                            Geom::tPoint pPhysics = vfv->GetCellQuadraturePPhys(iCell, ig);
                            inc = SpecialFields::IsentropicVortexCent(*this, pPhysics, 0, nVars);
                            inc *= vfv->GetCellJacobiDet(iCell, ig); // don't forget this
                        });
                    u[iCell] = um / vfv->GetCellVol(iCell); // mean value
                }
            break;
        case 3: // for taylor-green vortex problem
            DNDS_assert(model == NS_3D);
            if constexpr (model == NS_3D)
            {
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    Geom::tPoint pos = vfv->GetCellBary(iCell);
                    real M0 = 0.1;
                    real gamma = settings.idealGasProperty.gamma;
                    auto c2n = mesh->cell2node[iCell];
                    auto gCell = vfv->GetCellQuad(iCell);
                    TU um;
                    um.resizeLike(u[iCell]);
                    um.setZero();
                    // Eigen::MatrixXd coords;
                    // mesh->GetCoords(c2n, coords);
                    gCell.IntegrationSimple(
                        um,
                        [&](TU &inc, int ig)
                        {
                            // std::cout << coords<< std::endl << std::endl;
                            // std::cout << DiNj << std::endl;
                            Geom::tPoint pPhysics = vfv->GetCellQuadraturePPhys(iCell, ig);
                            real x{pPhysics(0)}, y{pPhysics(1)}, z{pPhysics(2)};
                            real ux = std::sin(x) * std::cos(y) * std::cos(z);
                            real uy = -std::cos(x) * std::sin(y) * std::cos(z);
                            real p = 1. / (gamma * sqr(M0)) + 1. / 16 * ((std::cos(2 * x) + std::cos(2 * y)) * (2 + std::cos(2 * z)));
                            real rho = gamma * sqr(M0) * p;
                            real E = 0.5 * (sqr(ux) + sqr(uy)) * rho + p / (gamma - 1);

                            // std::cout << T << " " << rho << std::endl;
                            inc.setZero();
                            inc(0) = rho;
                            inc(1) = rho * ux;
                            inc(2) = rho * uy;
                            inc(dim + 1) = E;

                            inc *= vfv->GetCellJacobiDet(iCell, ig); // don't forget this
                        });
                    u[iCell] = um / vfv->GetCellVol(iCell); // mean value
                }
            }
            break;
        case 0:
            break;
        default:
            log() << "Wrong specialBuiltinInitializer" << std::endl;
            DNDS_assert(false);
            break;
        }

        if (settings.frameConstRotation.enabled)
        {
            for (int iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                TU ui = u[iCell];
                TransformURotatingFrame(ui, vfv->GetCellQuadraturePPhys(iCell, -1), -1);
                u[iCell] = ui;
            }
        }

        // Box
        for (auto &i : settings.boxInitializers)
        {
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                Geom::tPoint pos = vfv->GetCellBary(iCell);
                if (pos(0) > i.x0 && pos(0) < i.x1 &&
                    pos(1) > i.y0 && pos(1) < i.y1 &&
                    pos(2) > i.z0 && pos(2) < i.z1)
                {
                    u[iCell] = i.v;
                }
            }
        }

        // Plane
        for (auto &i : settings.planeInitializers)
        {
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                Geom::tPoint pos = vfv->GetCellBary(iCell);
                if (pos(0) * i.a + pos(1) * i.b + pos(2) * i.c + i.h > 0)
                {
                    // std::cout << pos << std::endl << i.a << i.b << std::endl << i.h <<std::endl;
                    // DNDS_assert(false);
                    u[iCell] = i.v;
                }
            }
        }
    }

    template <EulerModel model>
    void EulerEvaluator<model>::FixUMaxFilter(ArrayDOFV<nVarsFixed> &u)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        // TODO: make spacial filter jacobian
        return; // ! nofix shortcut
    }

    template <EulerModel model>
    void EulerEvaluator<model>::TimeAverageAddition(ArrayDOFV<nVarsFixed> &w, ArrayDOFV<nVarsFixed> &wAveraged, real dt, real &tCur)
    {
        wAveraged *= (tCur / (tCur + dt + verySmallReal));
        wAveraged.addTo(w, (dt + verySmallReal) / (tCur + dt + verySmallReal));
        tCur += dt + verySmallReal;
    }

    template <EulerModel model>
    void EulerEvaluator<model>::MeanValueCons2Prim(ArrayDOFV<nVarsFixed> &u, ArrayDOFV<nVarsFixed> &w)
    {
        for (index iCell = 0; iCell < u.Size(); iCell++)
        {
            real gamma = settings.idealGasProperty.gamma;
            TU out;
            Gas::IdealGasThermalConservative2Primitive(u[iCell], out, gamma);
            w[iCell] = out;
        }
    }

    template <EulerModel model>
    void EulerEvaluator<model>::MeanValuePrim2Cons(ArrayDOFV<nVarsFixed> &w, ArrayDOFV<nVarsFixed> &u)
    {
        for (index iCell = 0; iCell < w.Size(); iCell++)
        {
            real gamma = settings.idealGasProperty.gamma;
            TU out;
            Gas::IdealGasThermalPrimitive2Conservative(w[iCell], out, gamma);
            u[iCell] = out;
        }
    }

    template <EulerModel model>
    void EulerEvaluator<model>::EvaluateNorm(Eigen::Vector<real, -1> &res, ArrayDOFV<nVarsFixed> &rhs, index P, bool volWise)
    {
        res.resize(nVars);
        if (P < 3)
        {
            TU resc;
            resc.setZero(nVars);

            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                if (rhs[iCell].hasNaN() || (!rhs[iCell].allFinite()))
                {
                    std::cout << rhs[iCell] << std::endl;
                    DNDS_assert(false);
                }
                if (volWise)
                    resc += rhs[iCell].array().abs().pow(P).matrix() * vfv->GetCellVol(iCell);
                else
                    resc += rhs[iCell].array().abs().pow(P).matrix();
            }
            MPI::Allreduce(resc.data(), res.data(), res.size(), DNDS_MPI_REAL, MPI_SUM, rhs.father->getMPI().comm);
            res = res.array().pow(1.0 / P).matrix();
            // std::cout << res << std::endl;
        }
        else
        {
            TU resc;
            resc.resizeLike(rhs[0]);
            resc.setZero();
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                resc = resc.array().max(rhs[iCell].array().abs()).matrix();
            MPI::Allreduce(resc.data(), res.data(), res.size(), DNDS_MPI_REAL, MPI_MAX, rhs.father->getMPI().comm);
        }
    }

    template <EulerModel model>
    void EulerEvaluator<model>::EvaluateRecNorm(
        Eigen::Vector<real, -1> &res,
        ArrayDOFV<nVarsFixed> &u,
        ArrayRECV<nVarsFixed> &uRec,
        index P,
        bool compare,
        const tFCompareField &FCompareField,
        const tFCompareFieldWeight &FCompareFieldWeight,
        real t)
    {
        res.resize(nVars);
        TU resc;
        resc.setZero(nVars);
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            auto qCell = vfv->GetCellQuad(iCell);
            TU rescCell;
            rescCell.setZero(nVars);
            qCell.IntegrationSimple(
                rescCell,
                [&](auto &inc, int iG)
                {
                    TU uR = u[iCell] + (vfv->GetIntPointDiffBaseValue(iCell, -1, -1, iG, 0, 1) * uRec[iCell]).transpose();
                    if (compare)
                    {
                        Geom::tPoint pPhysics = vfv->GetCellQuadraturePPhys(iCell, iG);
                        uR -= FCompareField(pPhysics, t);
                        uR *= FCompareFieldWeight(pPhysics, t);
                    }
                    if (P >= 3)
                        resc = resc.array().min(inc.array());
                    inc = uR.array().abs().pow(P);
                    inc *= vfv->GetCellJacobiDet(iCell, iG);
                });
            if (P < 3)
                resc += rescCell;
        }
        if (P > 3)
            MPI::Allreduce(resc.data(), res.data(), res.size(), DNDS_MPI_REAL, MPI_MAX, u.father->getMPI().comm);
        else
        {
            MPI::Allreduce(resc.data(), res.data(), res.size(), DNDS_MPI_REAL, MPI_SUM, u.father->getMPI().comm);
            res = res.array().pow(1.0 / P).matrix();
        }
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>, template <>
    )
    void EulerEvaluator<model>::EvaluateURecBeta(
        ArrayDOFV<nVarsFixed> &u,
        ArrayRECV<nVarsFixed> &uRec,
        ArrayDOFV<1> &uRecBeta, index &nLim, real &betaMin,
        int flag)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        real rhoEps = smallReal * settings.refUPrim(0);
        real pEps = smallReal * settings.refUPrim(I4);

        index nLimLocal = 0;
        real minBetaLocal = 1;
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            auto gCell = vfv->GetCellQuad(iCell);
            int nPoint = gCell.GetNumPoints();
            auto c2f = mesh->cell2face[iCell];
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                auto gFace = vfv->GetFaceQuad(c2f[ic2f]);
                nPoint += gFace.GetNumPoints();
            }
            /***********/
            Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> quadBase;
            quadBase.resize(nPoint, vfv->GetCellAtr(iCell).NDOF - 1);
            for (int iG = 0; iG < gCell.GetNumPoints(); iG++)
                quadBase(iG, Eigen::all) = vfv->GetIntPointDiffBaseValue(iCell, -1, -1, iG, 0, 1);
            nPoint = gCell.GetNumPoints();
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                auto gFace = vfv->GetFaceQuad(c2f[ic2f]);
                for (int iG = 0; iG < gFace.GetNumPoints(); iG++)
                    quadBase(nPoint + iG, Eigen::all) = vfv->GetIntPointDiffBaseValue(iCell, c2f[ic2f], -1, iG, 0, 1);
                nPoint += gFace.GetNumPoints();
            }
            /***********/
            DNDS_assert(u[iCell](0) >= rhoEps);
            real gamma = settings.idealGasProperty.gamma;
            real pCent = (gamma - 1) * (u[iCell](I4) - 0.5 * u[iCell](Seq123).squaredNorm() / u[iCell](0));
            DNDS_assert(pCent >= pEps);

            // alter uRec if necessary
            int curOrder = vfv->GetCellOrder(iCell);
            Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> uRecBase = uRec[iCell];
            Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> recBase; // * has to call checkRecBaseGood() to hold valid value
            auto checkRecBaseGood = [&]()
            {
                recBase = (quadBase * uRecBase).rowwise() + u[iCell].transpose();
                if (recBase(Eigen::all, 0).minCoeff() < rhoEps) // TODO: add relaxation to eps values
                    return false;
                if constexpr (model == NS_SA || model == NS_SA_3D)
                    if (recBase(Eigen::all, I4 + 1).minCoeff() < rhoEps)
                        return false;
                if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
                    if (recBase(Eigen::all, I4 + 1).minCoeff() < rhoEps || recBase(Eigen::all, I4 + 2).minCoeff() < rhoEps)
                        return false;
                Eigen::Vector<real, Eigen::Dynamic> ek =
                    0.5 * (recBase(Eigen::all, Seq123).array().square().rowwise().sum()) / recBase(Eigen::all, 0).array();
                Eigen::Vector<real, Eigen::Dynamic> eInternalS = (recBase(Eigen::all, I4) - ek);
                if (eInternalS.minCoeff() < pEps)
                    return false;
                return true;
            };
            if (checkRecBaseGood())
            {
                uRecBeta[iCell](0) = 1;
                continue; // early exit, reconstruction is good it self
            }
            if (flag == 1)
                curOrder = 1;
            while (curOrder > 0)
            {
                uRecBase = vfv->template DownCastURecOrder<nVarsFixed>(curOrder, iCell, uRec, 0);
                if (checkRecBaseGood())
                    break;
                uRec[iCell] = uRecBase; // uRec[iCell] could be altered
                curOrder--;
            }

            Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed>
                recInc = quadBase * (uRec[iCell] - uRecBase);
            Eigen::Vector<real, Eigen::Dynamic> rhoS = recInc(Eigen::all, 0) + recBase(Eigen::all, 0);
            Eigen::Index rhoMinIdx;
            real rhoMin = rhoS.minCoeff(&rhoMinIdx);
            real theta1 = 1;
            if (rhoMin < rhoEps)
                for (int iG = 0; iG < rhoS.size(); iG++)
                    if (recInc(iG, 0) < 0) // negative increment
                        theta1 = std::min(theta1, (recBase(iG, 0) - rhoEps) / (-recInc(iG, 0) + verySmallReal));
#ifdef USE_NS_SA_NUT_REDUCED_ORDER
            if constexpr (model == NS_SA || model == NS_SA_3D)
            {
                static real v1Eps = smallReal * settings.refUPrim(I4 + 1);
                Eigen::Vector<real, Eigen::Dynamic> v1S = recInc(Eigen::all, I4 + 1) + recBase(Eigen::all, I4 + 1);
                real v1Min = v1S.minCoeff();
                if (v1Min < v1Eps)
                    for (int iG = 0; iG < rhoS.size(); iG++)
                        if (recInc(iG, I4 + 1) < 0) // negative increment
                            theta1 = std::min(theta1, (recBase(iG, I4 + 1) - v1Eps) / (-recInc(iG, I4 + 1) + verySmallReal))
                                // * 0 // to gain fully reduced order
                                ;
            }
#endif
            if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
            {

                static real v1Eps = smallReal * settings.refUPrim(I4 + 1);
                static real v2Eps = smallReal * settings.refUPrim(I4 + 2);
                Eigen::Vector<real, Eigen::Dynamic> v1S = recInc(Eigen::all, I4 + 1) + recBase(Eigen::all, I4 + 1);
                Eigen::Vector<real, Eigen::Dynamic> v2S = recInc(Eigen::all, I4 + 2) + recBase(Eigen::all, I4 + 2);
                real v1Min = v1S.minCoeff();
                real v2Min = v2S.minCoeff();
                if (v1Min < v1Eps)
                    for (int iG = 0; iG < rhoS.size(); iG++)
                        if (recInc(iG, I4 + 1) < 0) // negative increment
                            theta1 = std::min(theta1, (recBase(iG, I4 + 1) - v1Eps) / (-recInc(iG, I4 + 1) + verySmallReal));
                if (v2Min < v2Eps)
                    for (int iG = 0; iG < rhoS.size(); iG++)
                        if (recInc(iG, I4 + 2) < 0) // negative increment
                            theta1 = std::min(theta1, (recBase(iG, I4 + 2) - v2Eps) / (-recInc(iG, I4 + 2) + verySmallReal));
            }

            recInc *= theta1;
            Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed>
                recVRhoG = recInc + recBase;

            Eigen::Vector<real, Eigen::Dynamic> ek = 0.5 * (recVRhoG(Eigen::all, Seq123).array().square().rowwise().sum()) / recVRhoG(Eigen::all, 0).array();
            Eigen::Vector<real, Eigen::Dynamic> eInternalS = recVRhoG(Eigen::all, I4) - ek;
            real thetaP = 1.0;

            if (pCent <= 2 * pEps)
                thetaP = 0;
            else
                for (int iG = 0; iG < rhoS.size(); iG++)
                {
                    if (eInternalS(iG) < 2 * pEps / (gamma - 1))
                    {
                        real thetaThis = Gas::IdealGasGetCompressionRatioPressure<dim, 0, nVarsFixed>(
                            recBase(iG, Eigen::all).transpose(), recInc(iG, Eigen::all).transpose(),
                            pEps);
                        thetaP = std::min(thetaP, thetaThis);
                    }
                }

            uRecBeta[iCell](0) = theta1 * thetaP;
            // if (uRecBeta[iCell](0) < 1)
            //     uRecBeta[iCell](0) *= 1 - 1e-8;
            // uRecBeta[iCell](0) = std::min(theta1, thetaP);
            if (uRecBeta[iCell](0) < 1)
            {
                nLimLocal++;
                // uRecBeta[iCell](0) *= uRecBeta[iCell](0) < 0.99 ? 0. : 0.99; //! for safety
                uRecBeta[iCell](0) *= std::pow(uRecBeta[iCell](0), static_cast<int>(std::round(settings.uRecBetaCompressPower))) * 0.99;
                minBetaLocal = std::min(uRecBeta[iCell](0), minBetaLocal);
            }
            if (uRecBeta[iCell](0) < 0)
            {
                std::cout << fmt::format("theta1 {}, thetaP {}", theta1, thetaP) << std::endl;
                DNDS_assert(false);
            }
            if (uRecBeta[iCell](0) < 1)
                uRec[iCell] = (uRec[iCell] - uRecBase) * uRecBeta[iCell](0) + uRecBase;

            // validation:
            recInc = quadBase * uRec[iCell];
            recVRhoG = recInc.rowwise() + u[iCell].transpose();
            ek = 0.5 * (recVRhoG(Eigen::all, Seq123).array().square().rowwise().sum()) / recVRhoG(Eigen::all, 0).array();
            eInternalS = (recVRhoG(Eigen::all, I4) - ek);
            for (int iG = 0; iG < eInternalS.size(); iG++)
            {
                if (eInternalS(iG) < pEps)
                {
                    std::cout << std::scientific;
                    std::cout << eInternalS.transpose() << std::endl;
                    std::cout << curOrder << std::endl;
                    std::cout << fmt::format("{} {} {}", theta1, thetaP, uRecBeta[iCell](0)) << std::endl;
                    std::cout << u[iCell] << std::endl;
                    std::cout << recInc.transpose() << std::endl;
                    DNDS_assert(false);
                }
            }
        }
        MPI::Allreduce(&nLimLocal, &nLim, 1, DNDS_MPI_INDEX, MPI_SUM, u.father->getMPI().comm);
        MPI::Allreduce(&minBetaLocal, &betaMin, 1, DNDS_MPI_REAL, MPI_MIN, u.father->getMPI().comm);
    }

    template <EulerModel model>
    bool EulerEvaluator<model>::AssertMeanValuePP(
        ArrayDOFV<nVarsFixed> &u, bool panic)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        real rhoEps = smallReal * settings.refUPrim(0);
        real pEps = smallReal * settings.refUPrim(I4);

        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            real gamma = settings.idealGasProperty.gamma;
            real alphaRho = 1;
            if (u[iCell](0) < rhoEps)
            {
                if (panic)
                    DNDS_assert_info(
                        false,
                        fmt::format(
                            "AssertMeanValuePP Failed on cell {} rho\n",
                            iCell) +
                            fmt::format(
                                " eps={}, value={}",
                                rhoEps, u[iCell](0)));
                return false;
            }
            real rhoEi = u[iCell](I4) - 0.5 * u[iCell](Seq123).squaredNorm() / u[iCell](0);
            if (rhoEi < pEps / (gamma - 1))
            {
                if (panic)
                    DNDS_assert_info(
                        false,
                        fmt::format(
                            "AssertMeanValuePP Failed on cell {} rhoEi\n",
                            iCell) +
                            fmt::format(
                                " eps={}, value={}",
                                pEps / (gamma - 1), rhoEi));
                return false;
            }
        }

        return true;
    }

    template <EulerModel model>
    void EulerEvaluator<model>::EvaluateCellRHSAlpha(
        ArrayDOFV<nVarsFixed> &u,
        ArrayRECV<nVarsFixed> &uRec,
        ArrayDOFV<1> &uRecBeta,
        ArrayDOFV<nVarsFixed> &res,
        ArrayDOFV<1> &cellRHSAlpha, index &nLim, real &alphaMin, real relax,
        int flag)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        real rhoEps = smallReal * settings.refUPrim(0);
        real pEps = smallReal * settings.refUPrim(I4);

        index nLimLocal = 0;
        real alphaMinLocal = 1;
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            real gamma = settings.idealGasProperty.gamma;
            real alphaRho = 1;
            TU inc = res[iCell];
            DNDS_assert(u[iCell](0) >= rhoEps);
            real relaxedRho = rhoEps + (u[iCell](0) - rhoEps) * (1 - relax);
            if (inc(0) < 0) // not < rhoEps!!!
                alphaRho = std::min(1.0, (u[iCell](0) - relaxedRho) / (-inc(0) - smallReal * inc(0)));
            DNDS_assert(alphaRho >= 0 && alphaRho <= 1);
            if constexpr (model == NS_SA || model == NS_SA_3D)
            {
                // ** ! currently do not mass - fix for SA
                // static real v1Eps = smallReal * settings.refUPrim(I4 + 1);
                // if (inc(I4 + 1) < 0)
                //     alphaRho = std::min(alphaRho,
                //                         (u[iCell](I4 + 1) - v1Eps) / (-inc(I4 + 1) - smallReal * inc(I4 + 1)));
                // // use exp down:
                // if (inc(I4 + 1) + u[iCell](I4 + 1) < v1Eps)
                // {
                //     DNDS_assert(inc(I4 + 1) < 0);
                //     real declineV = inc(I4 + 1) / (u[iCell](I4 + 1) + 1e-6);
                //     real newu5 = u[iCell](I4 + 1) * std::exp(declineV);
                //     alphaRho = std::min(alphaRho,
                //                         (u[iCell](I4 + 1) - newu5) / (-inc(I4 + 1) - smallReal * inc(I4 + 1)));
                // }
            }
            if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
            {
                static real v1Eps = smallReal * settings.refUPrim(I4 + 1);
                static real v2Eps = smallReal * settings.refUPrim(I4 + 2);
                if (inc(I4 + 1) < 0)
                    alphaRho = std::min(alphaRho,
                                        (u[iCell](I4 + 1) - v1Eps) / (-inc(I4 + 1) - smallReal * inc(I4 + 1)));
                if (inc(I4 + 2) < 0)
                    alphaRho = std::min(alphaRho,
                                        (u[iCell](I4 + 2) - v2Eps) / (-inc(I4 + 2) - smallReal * inc(I4 + 2)));
            }

            inc *= alphaRho;

            TU uNew = u[iCell] + inc;
            real pNew = (uNew(I4) - 0.5 * uNew(Seq123).squaredNorm() / uNew(0)) * (gamma - 1);
            real pOld = (u[iCell](I4) - 0.5 * u[iCell](Seq123).squaredNorm() / u[iCell](0)) * (gamma - 1);
            real relaxedP = pEps;
            if (pNew < pOld)
                relaxedP = pEps + (pOld - pEps) * (1 - relax);

            real alphaP = 1;
            if (pNew < relaxedP)
            {
                // todo: use high order accurate (add control switch)
                real alphaC = Gas::IdealGasGetCompressionRatioPressure<dim, 0, nVarsFixed>(
                    u[iCell], inc, relaxedP / (gamma - 1));
                alphaP = std::min(alphaP, alphaC);
            }
            cellRHSAlpha[iCell](0) = alphaRho * alphaP;
            // cellRHSAlpha[iCell](0) = std::min(alphaRho, alphaP);
            if (cellRHSAlpha[iCell](0) < 1)
            {
                cellRHSAlpha[iCell](0) = std::pow(cellRHSAlpha[iCell](0), static_cast<int>(std::round(settings.uRecAlphaCompressPower)));
                nLimLocal++,
                    cellRHSAlpha[iCell] *= (0.9),
                    alphaMinLocal = std::min(alphaMinLocal, cellRHSAlpha[iCell](0));
            } //! for safety
        }
        MPI::Allreduce(&nLimLocal, &nLim, 1, DNDS_MPI_INDEX, MPI_SUM, u.father->getMPI().comm);
        MPI::Allreduce(&alphaMinLocal, &alphaMin, 1, DNDS_MPI_REAL, MPI_MIN, u.father->getMPI().comm);
        if (flag == 0)
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                if (cellRHSAlpha[iCell](0) < 1)
                    cellRHSAlpha[iCell](0) = alphaMin;
            }
        if (flag == -1)
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                cellRHSAlpha[iCell](0) = alphaMin;
    }

    template <EulerModel model>
    void EulerEvaluator<model>::EvaluateCellRHSAlphaExpansion(
        ArrayDOFV<nVarsFixed> &u,
        ArrayRECV<nVarsFixed> &uRec,
        ArrayDOFV<1> &uRecBeta,
        ArrayDOFV<nVarsFixed> &res,
        ArrayDOFV<1> &cellRHSAlpha, index &nLim, real alphaMin)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        real rhoEps = smallReal * settings.refUPrim(0);
        real pEps = smallReal * settings.refUPrim(I4);

        auto cellIsHalfAlpha = [&](index iCell) -> bool // iCell should be internal
        {
            bool ret = false;
            if (cellRHSAlpha[iCell](0) == 1.0)
            {
                auto c2f = mesh->cell2face[iCell];
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iCellOther = vfv->CellFaceOther(iCell, c2f[ic2f]);
                    if (iCellOther != UnInitIndex)
                        if (cellRHSAlpha[iCellOther](0) != 1.0)
                            ret = true;
                }
            }
            return ret;
        };

        auto cellAdjAlphaMin = [&](index iCell) -> real // iCell should be internal
        {
            real ret = 1;
            auto c2f = mesh->cell2face[iCell];
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iCellOther = vfv->CellFaceOther(iCell, c2f[ic2f]);
                if (iCellOther != UnInitIndex)
                    ret = std::min(ret, cellRHSAlpha[iCellOther](0));
            }
            return ret;
        };

        // std::vector<index> InterCells;

        // for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        //     if (cellIsHalfAlpha(iCell))
        //         InterCells.emplace_back(iCell);

        index nLimLocal = 0;
        index nLimAdd = 0;
        // for (index iCell : InterCells)
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            real gamma = settings.idealGasProperty.gamma;
            TU inc = res[iCell];

            TU uNew = u[iCell] + inc;
            real pNew = (uNew(I4) - 0.5 * uNew(Seq123).squaredNorm() / uNew(0)) * (gamma - 1);

            if (pNew < pEps || uNew(0) < rhoEps)
            {
                // cellRHSAlpha[iCell](0) = cellAdjAlphaMin(iCell);
                // DNDS_assert(cellRHSAlpha[iCell](0) == alphaMin);
                cellRHSAlpha[iCell](0) = alphaMin;
            }
            if constexpr (model == NS_SA || model == NS_SA_3D)
            {
                // ** ! currently do not mass - fix for SA
                // static real v1Eps = smallReal * settings.refUPrim(I4 + 1);
                // if (uNew(I4 + 1) < v1Eps)
                //     cellRHSAlpha[iCell](0) = alphaMin;
            }
            if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
            {
                static real v1Eps = smallReal * settings.refUPrim(I4 + 1);
                static real v2Eps = smallReal * settings.refUPrim(I4 + 2);
                if (uNew(I4 + 1) < v1Eps)
                    cellRHSAlpha[iCell](0) = alphaMin;
                if (uNew(I4 + 2) < v2Eps)
                    cellRHSAlpha[iCell](0) = alphaMin;
            }
        }
        MPI::Allreduce(&nLimLocal, &nLimAdd, 1, DNDS_MPI_INDEX, MPI_SUM, u.father->getMPI().comm);
        nLim += nLimAdd;
    }

    template <EulerModel model>
    void EulerEvaluator<model>::MinSmoothDTau(
        ArrayDOFV<1> &dTau, ArrayDOFV<1> &dTauNew)
    {
        real smootherCentWeight = 1;
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            auto c2f = mesh->cell2face[iCell];
            real nAdj = 0.;
            real dTMean = 0.;
            for (index iFace : c2f)
            {
                index iCellOther = vfv->CellFaceOther(iCell, iFace);
                if (iCellOther != UnInitIndex)
                {
                    nAdj += 1.;
                    dTMean += dTau[iCellOther](0);
                }
            }
            dTMean += nAdj * smootherCentWeight * dTau[iCell](0);
            dTMean /= nAdj * (1 + smootherCentWeight);
            dTauNew[iCell](0) = std::min(dTau[iCell](0), dTMean);
        }
    }
}
