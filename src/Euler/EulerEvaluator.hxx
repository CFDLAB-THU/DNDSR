#pragma once

#include "DNDS/Defines.hpp" // for correct  DNDS_SWITCH_INTELLISENSE
#include "EulerEvaluator.hpp"
#include "DNDS/HardEigen.hpp"
#include "SpecialFields.hpp"
#include "DNDS/ExprtkWrapper.hpp"

namespace DNDS::Euler
{

    static const auto model = NS_SA; // to be hidden by template params

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>, )
    void EulerEvaluator<model>::LUSGSMatrixInit(
        JacobianDiagBlock<nVarsFixed> &JDiag,
        JacobianDiagBlock<nVarsFixed> &JSource,
        ArrayDOFV<1> &dTau, real dt, real alphaDiag,
        ArrayDOFV<nVarsFixed> &u,
        ArrayRECV<nVarsFixed> &uRec,
        int jacobianCode,
        real t)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        // TODO: for code0: flux jacobian with lambdaFace, and source jacobian with integration, only diagpart dealt with
        DNDS_assert(JDiag.isBlock() == JSource.isBlock());
        DNDS_assert(jacobianCode == 0);
        if (settings.useRoeJacobian)
            DNDS_assert(JDiag.isBlock());
        JDiag.clearValues();
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
                if (!settings.useRoeJacobian)
                    continue;
                // roe term jacobi
                auto f2c = mesh->face2cell[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                rowsize iCellAtFace = f2c[0] == iCell ? 0 : 1;
                TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCell, iCellAtFace, -1)(Seq012) *
                                (iCellAtFace ? -1 : 1); // faces out
                if (iCellOther == UnInitIndex)
                    iCellOther = iCell; //! todo: deal with bcs
                TU uj = u[iCellOther];
                if (iCellOther != UnInitIndex)
                    this->UFromOtherCell(uj, iFace, iCell, iCellOther, iCellAtFace);
                TJacobianU jacII = fluxJacobian0_Right_Times_du_AsMatrix( // unitnorm and uj are both respect with this cell
                    u[iCell], uj,
                    unitNorm, GetFaceVGridFromCell(iFace, iCell, iCellAtFace, -1),
                    Geom::BC_ID_INTERNAL,
                    lambdaFace[iFace], lambdaFaceC[iFace], lambdaFaceVis[iFace],
                    iCellAtFace ? lambdaFace4[iFace] : lambdaFace0[iFace], lambdaFace123[iFace], iCellAtFace ? lambdaFace0[iFace] : lambdaFace4[iFace],
                    // swap lambda0 and lambda4 if iCellAtFace==1
                    true, +1, 1); // for this is diff(uthis) not diff(uthat)
                JDiag.getBlock(iCell) += (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) * jacII;
                // std::cout << "JacII\n";
                // std::cout << jacII << "\n";
                // std::cout << lambdaFace[iFace] << std::endl;
            }
            if (settings.useRoeJacobian)
                JDiag.getBlock(iCell).diagonal().array() += 1.0 / dTau[iCell](0) + 1.0 / dt; // time term!!
            else
            {
                if (JDiag.isBlock())
                    JDiag.getBlock(iCell).diagonal().setConstant(fpDivisor);
                else
                    JDiag.getDiag(iCell).setConstant(fpDivisor);
            }

            // std::cout << fpDivisor << std::endl;

            // jacobian diag

            if (!settings.ignoreSourceTerm)
            {
                if (JDiag.isBlock())
                    JDiag.getBlock(iCell) += alphaDiag * JSource.getBlock(iCell);
                else
                    JDiag.getDiag(iCell) += alphaDiag * JSource.getDiag(iCell);
            }

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
        JacobianDiagBlock<nVarsFixed> &JDiag,
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
                TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCell, iCellAtFace, -1)(Seq012) *
                                (iCellAtFace ? -1 : 1); // faces out
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

                            // fInc = fluxJacobian0_Right(
                            //            u[iCellOther],
                            //            unitNorm,
                            //            BoundaryType::Inner) *
                            //        uInc[iCellOther]; //! always inner here
                            fInc = fluxJacobian0_Right_Times_du(
                                uj, u[iCell],
                                unitNorm, GetFaceVGridFromCell(iFace, iCell, iCellAtFace, -1),
                                Geom::BC_ID_INTERNAL, uINCj,
                                lambdaFace[iFace], lambdaFaceC[iFace], lambdaFaceVis[iFace],
                                iCellAtFace ? lambdaFace4[iFace] : lambdaFace0[iFace], lambdaFace123[iFace], iCellAtFace ? lambdaFace0[iFace] : lambdaFace4[iFace],
                                // swap lambda0 and lambda4 if iCellAtFace==1
                                settings.useRoeJacobian); //! always inner here
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
            AuIncI = JDiag.MatVecLeft(iCell, uInc[iCell]) - uIncNewBuf;

            if (AuIncI.hasNaN())
            {
                std::cout << AuIncI.transpose() << std::endl
                          << uINCi.transpose() << std::endl
                          << u[iCell].transpose() << std::endl
                          << JDiag.getValue(iCell) << std::endl
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
        JacobianDiagBlock<nVarsFixed> &JDiag,
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
                TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCell, iCellAtFace, -1)(Seq012) *
                                (iCellAtFace ? -1 : 1); // faces out
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
                        jacIJ = fluxJacobian0_Right_Times_du_AsMatrix( // unitnorm and uj are both respect with this cell
                            uj, u[iCell],
                            unitNorm, GetFaceVGridFromCell(iFace, iCell, iCellAtFace, -1),
                            Geom::BC_ID_INTERNAL,
                            lambdaFace[iFace], lambdaFaceC[iFace], lambdaFaceVis[iFace],
                            iCellAtFace ? lambdaFace4[iFace] : lambdaFace0[iFace], lambdaFace123[iFace], iCellAtFace ? lambdaFace0[iFace] : lambdaFace4[iFace],
                            // swap lambda0 and lambda4 if iCellAtFace==1
                            settings.useRoeJacobian); //! always inner here
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
            jacLU.GetDiag(iCell) = JDiag.getValue(iCell);
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
        JacobianDiagBlock<nVarsFixed> &JDiag,
        ArrayDOFV<nVarsFixed> &uIncNew)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateLUSGSForward 1");
        int cnvars = nVars;
        JDiag.GetInvert();
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
                TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCell, iCellAtFace, -1)(Seq012) *
                                (iCellAtFace ? -1 : 1); // faces out
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
                            fInc = fluxJacobian0_Right_Times_du(
                                uj, u[iCell],
                                unitNorm, GetFaceVGridFromCell(iFace, iCell, iCellAtFace, -1),
                                Geom::BC_ID_INTERNAL, uINCj,
                                lambdaFace[iFace], lambdaFaceC[iFace], lambdaFaceVis[iFace],
                                iCellAtFace ? lambdaFace4[iFace] : lambdaFace0[iFace], lambdaFace123[iFace], iCellAtFace ? lambdaFace0[iFace] : lambdaFace4[iFace],
                                // swap lambda0 and lambda4 if iCellAtFace==1
                                settings.useRoeJacobian); //! always inner here
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
            uIncNewI = JDiag.MatVecLeftInvert(iCell, uIncNewBuf);

            if (uIncNewI.hasNaN())
            {
                std::cout << uIncNewI.transpose() << std::endl
                          << uIncNewBuf.transpose() << std::endl
                          << JDiag.getValue(iCell) << std::endl
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
        JacobianDiagBlock<nVarsFixed> &JDiag,
        ArrayDOFV<nVarsFixed> &uIncNew)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateLUSGSBackward 1");
        int cnvars = nVars;
        JDiag.GetInvert();
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
                TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCell, iCellAtFace, -1)(Seq012) *
                                (iCellAtFace ? -1 : 1); // faces out
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

                            fInc = fluxJacobian0_Right_Times_du(
                                uj, u[iCell],
                                unitNorm, GetFaceVGridFromCell(iFace, iCell, iCellAtFace, -1),
                                Geom::BC_ID_INTERNAL, uINCj,
                                lambdaFace[iFace], lambdaFaceC[iFace], lambdaFaceVis[iFace],
                                iCellAtFace ? lambdaFace4[iFace] : lambdaFace0[iFace], lambdaFace123[iFace], iCellAtFace ? lambdaFace0[iFace] : lambdaFace4[iFace],
                                // swap lambda0 and lambda4 if iCellAtFace==1
                                settings.useRoeJacobian); //! always inner here
                        }

                        uIncNewBuf -= (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) *
                                      (fInc);
                    }
                }
            }
            auto uIncNewI = uIncNew[iCell];
            uIncNewI += JDiag.MatVecLeftInvert(iCell, uIncNewBuf);
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
        JacobianDiagBlock<nVarsFixed> &JDiag,
        bool forward, TU &sumInc)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateSGS 1");
        int cnvars = nVars;
        JDiag.GetInvert();
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
                auto btype = mesh->GetFaceZone(iFace);
                auto f2c = mesh->face2cell[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCell, iCellAtFace, -1)(Seq012) *
                                (iCellAtFace ? -1 : 1); // faces out
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
                            fInc = fluxJacobian0_Right_Times_du(
                                uj, u[iCell],
                                unitNorm, GetFaceVGridFromCell(iFace, iCell, iCellAtFace, -1),
                                Geom::BC_ID_INTERNAL, uINCj,
                                lambdaFace[iFace], lambdaFaceC[iFace], lambdaFaceVis[iFace],
                                iCellAtFace ? lambdaFace4[iFace] : lambdaFace0[iFace], lambdaFace123[iFace], iCellAtFace ? lambdaFace0[iFace] : lambdaFace4[iFace],
                                // swap lambda0 and lambda4 if iCellAtFace==1
                                settings.useRoeJacobian); //! always inner here
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
                // else if (pBCHandler->GetTypeFromID(btype) == BCWall)
                // {
                //     TMat normBase = Geom::NormBuildLocalBaseV<dim>(unitNorm);
                //     Geom::tPoint pPhysics = vfv->GetFaceQuadraturePPhysFromCell(iFace, iCell, iCellAtFace, -1);
                //     TU uThis = u[iCell];
                //     TU uINCj = uInc[iCell];
                //     //! using t = 0 in generateBoudnaryValue!
                //     TU uj = generateBoundaryValue(uThis, uThis, iCell, iFace, -1, unitNorm, normBase, pPhysics, 0, btype, false, 0);
                //     uINCj(Seq123) *= -1;

                //     if (model == NS_SA || model == NS_SA_3D)
                //         uINCj(I4 + 1) = 0;
                //     if (model == NS_2EQ || model == NS_2EQ_3D)
                //         uINCj({I4 + 1, I4 + 2}).setZero();
                //     TU fInc = fluxJacobian0_Right_Times_du(
                //         uj, u[iCell],
                //         unitNorm, GetFaceVGridFromCell(iFace, iCell, iCellAtFace, -1),
                //         Geom::BC_ID_INTERNAL, uINCj,
                //         lambdaFace[iFace], lambdaFaceC[iFace], lambdaFaceVis[iFace],
                //         iCellAtFace ? lambdaFace4[iFace] : lambdaFace0[iFace], lambdaFace123[iFace], iCellAtFace ? lambdaFace0[iFace] : lambdaFace4[iFace],
                //                // swap lambda0 and lambda4 if iCellAtFace==1
                //         settings.useRoeJacobian); //! treat as inner here

                //     uIncNewBuf -= (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) *
                //                   (fInc);
                // }
            }
            auto uIncNewI = uIncNew[iCell];
            TU uIncOld = uIncNewI;

            uIncNewI = JDiag.MatVecLeftInvert(iCell, uIncNewBuf);
            sumInc.array() += (uIncNewI - uIncOld).array().abs();

            if (uIncNewI.hasNaN())
            {
                std::cout << uIncNewI.transpose() << std::endl
                          << uIncNewBuf.transpose() << std::endl
                          << JDiag.getValue(iCell) << std::endl
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
        JacobianDiagBlock<nVarsFixed> &JDiag,
        bool forward, TU &sumInc)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateSGS 1");
        int cnvars = nVars;
        JDiag.GetInvert();
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
                TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCell, iCellAtFace, -1)(Seq012) *
                                (iCellAtFace ? -1 : 1); // faces out
                if (iCellOther != UnInitIndex)
                {
                    index iScanOther = forward ? iCellOther : nCellDist - 1 - iCellOther; // TODO: add rb-sor
                    if (iCell != iCellOther)
                    {
                        TU fInc, fIncS;
                        auto uINCj = uInc[iCellOther];
                        {
                            fInc = fluxJacobian0_Right_Times_du(
                                u[iCellOther], u[iCell], //! TODO periodic here
                                unitNorm, GetFaceVGridFromCell(iFace, iCell, iCellAtFace, -1),
                                Geom::BC_ID_INTERNAL, uINCj,
                                lambdaFace[iFace], lambdaFaceC[iFace], lambdaFaceVis[iFace],
                                iCellAtFace ? lambdaFace4[iFace] : lambdaFace0[iFace], lambdaFace123[iFace], iCellAtFace ? lambdaFace0[iFace] : lambdaFace4[iFace],
                                // swap lambda0 and lambda4 if iCellAtFace==1
                                settings.useRoeJacobian); //! always inner here
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
                            TU fIncSL = fluxJacobianC_Right_Times_du(u[iCell], unitNorm, GetFaceVGridFromCell(iFace, iCell, iCellAtFace, -1), Geom::BC_ID_INTERNAL, uRecSLInc);
                            TU fIncSR = fluxJacobianC_Right_Times_du(u[iCellOther], unitNorm, GetFaceVGridFromCell(iFace, iCell, iCellAtFace, -1), Geom::BC_ID_INTERNAL, uRecSRInc);
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

            uIncNewI = JDiag.MatVecLeftInvert(iCell, uIncNewBuf);
            sumInc.array() += (uIncNewI - uIncOld).array().abs();

            if (uIncNewI.hasNaN())
            {
                std::cout << uIncNewI.transpose() << std::endl
                          << uIncNewBuf.transpose() << std::endl
                          << JDiag.getValue(iCell) << std::endl
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
        JacobianDiagBlock<nVarsFixed> &JDiag,
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
                TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCell, iCellAtFace, -1)(Seq012) *
                                (iCellAtFace ? -1 : 1); // faces out
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

                        fInc = fluxJacobian0_Right_Times_du(
                            uj, u[iCell],
                            unitNorm, GetFaceVGridFromCell(iFace, iCell, iCellAtFace, -1),
                            Geom::BC_ID_INTERNAL, uINCj,
                            lambdaFace[iFace], lambdaFaceC[iFace], lambdaFaceVis[iFace],
                            iCellAtFace ? lambdaFace4[iFace] : lambdaFace0[iFace], lambdaFace123[iFace], iCellAtFace ? lambdaFace0[iFace] : lambdaFace4[iFace],
                            // swap lambda0 and lambda4 if iCellAtFace==1
                            settings.useRoeJacobian); //! always inner here
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

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>, )
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
        if (model == EulerModel::NS_2EQ || model == NS_2EQ_3D)
        {
            if (settings.ransModel == RANSModel::RANS_KOSST ||
                settings.ransModel == RANSModel::RANS_KOWilcox)
                for (int iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    auto c2f = mesh->cell2face[iCell];
                    real d = dWall.at(iCell).mean();
                    // for SST or KOWilcox
                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                    // if (pBCHandler->GetTypeFromID(mesh->GetFaceZone(c2f[ic2f])) == EulerBCType::BCWall)
                    {
                        real pMean, asqrMean, Hmean;
                        real gamma = settings.idealGasProperty.gamma;
                        Gas::IdealGasThermal(u[iCell](I4), u[iCell](0), (u[iCell](Seq123) / u[iCell](0)).squaredNorm(),
                                             gamma, pMean, asqrMean, Hmean);
                        real muRef = settings.idealGasProperty.muGas;
                        real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * u[iCell](0));
                        real mufPhy1 = muEff(u[iCell], T);
                        real rhoOmegaaaWall = mufPhy1 / sqr(d) * 800 * 0.1;

                        real rhoOmegaaaNew = std::max(rhoOmegaaaWall, u[iCell](I4 + 2));
                        real rhoOmegaaaOld = u[iCell](I4 + 2);
                        // u[iCell](I4 + 2) = rhoOmegaaaNew;
                        // u[iCell](I4 + 1) = rhoOmegaaaNew / rhoOmegaaaOld * u[iCell](I4 + 1);
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
        case 2:   // for IV10 problem
        case 203: // for IV10 problem with PP
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
                                inc = SpecialFields::IsentropicVortex10(*this, pPhysics, 0, nVars, 5);
                            else if (settings.specialBuiltinInitializer == 203)
                                inc = SpecialFields::IsentropicVortex10(*this, pPhysics, 0, nVars, 10.0828);
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
        case 3001: // for nol problem
            DNDS_assert(model == NS_3D);
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
                        TU farPrimitive;
                        Gas::IdealGasThermalConservative2Primitive<dim>(settings.farFieldStaticValue, farPrimitive, settings.idealGasProperty.gamma);
                        real pInf = farPrimitive(I4);
                        real r = pPhysics.norm();
                        TVec velo = -pPhysics(Seq012) / (r + smallReal);
                        farPrimitive(I4) = pInf;
                        farPrimitive(0) = 1;
                        farPrimitive(Seq123) = velo;

                        Gas::IdealGasThermalPrimitive2Conservative<dim>(farPrimitive, inc, settings.idealGasProperty.gamma);
                        inc *= vfv->GetCellJacobiDet(iCell, ig); // don't forget this
                    });
                u[iCell] = um / vfv->GetCellVol(iCell); // mean value
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

        for (auto &i : settings.exprtkInitializers)
        {
            auto exprStr = i.GetExpr();
            ExprtkWrapperEvaluator exprtkEval;
            exprtkEval.AddScalar("inRegion");
            exprtkEval.AddScalar("iCell");
            exprtkEval.AddVector("x", dim);
            exprtkEval.AddVector("UPrim", nVars);
            exprtkEval.Compile(exprStr);
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                Geom::tPoint pos = vfv->GetCellBary(iCell);
                auto c2n = mesh->cell2node[iCell];
                auto gCell = vfv->GetCellQuad(iCell);
                TU um;
                um.resizeLike(u[iCell]);
                um.setZero();
                bool allIn = true;
                bool someIn = false;
                gCell.IntegrationSimple(
                    um,
                    [&](TU &inc, int ig)
                    {
                        Geom::tPoint pPhysics = vfv->GetCellQuadraturePPhys(iCell, ig);

                        exprtkEval.Var("inRegion") = 0;
                        exprtkEval.Var("iCell") = real(iCell);
                        exprtkEval.VarVec("x", 0) = pPhysics(0);
                        exprtkEval.VarVec("x", 1) = pPhysics(1);
                        if constexpr (dim == 3)
                            exprtkEval.VarVec("x", 2) = pPhysics(2);

                        TU uPrimitive;
                        Gas::IdealGasThermalConservative2Primitive<dim>(u[iCell], uPrimitive, settings.idealGasProperty.gamma);
                        for (int i = 0; i < nVars; i++)
                            exprtkEval.VarVec("UPrim", i) = uPrimitive(i);

                        real ret = exprtkEval.Evaluate();

                        DNDS_assert_info(ret == 0.0, "return of \n" + exprStr + "\nis non-zero");
                        allIn = allIn && exprtkEval.Var("inRegion");
                        someIn = someIn || exprtkEval.Var("inRegion");

                        if (exprtkEval.Var("inRegion"))
                            for (int i = 0; i < nVars; i++)
                                uPrimitive(i) = exprtkEval.VarVec("UPrim", i);
                        Gas::IdealGasThermalPrimitive2Conservative<dim>(uPrimitive, inc, settings.idealGasProperty.gamma);

                        inc *= vfv->GetCellJacobiDet(iCell, ig); // don't forget this
                    });
                // if (allIn)
                if (someIn)
                    u[iCell] = um / vfv->GetCellVol(iCell); // mean value
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
    void EulerEvaluator<model>::EvaluateNorm(Eigen::Vector<real, -1> &res, ArrayDOFV<nVarsFixed> &rhs, index P, bool volWise, bool average)
    {
        res.resize(nVars);
        if (P < 3)
        {
            TU resc;
            resc.setZero(nVars);
            real rescBase{0};

            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                if (rhs[iCell].hasNaN() || (!rhs[iCell].allFinite()))
                {
                    std::cout << rhs[iCell] << std::endl;
                    DNDS_assert(false);
                }
                if (volWise)
                    resc += rhs[iCell].array().abs().pow(P).matrix() * vfv->GetCellVol(iCell), rescBase += vfv->GetCellVol(iCell);
                else
                    resc += rhs[iCell].array().abs().pow(P).matrix(), rescBase += 1;
            }
            MPI::Allreduce(resc.data(), res.data(), res.size(), DNDS_MPI_REAL, MPI_SUM, rhs.father->getMPI().comm);
            if (average)
                MPI::AllreduceOneReal(rescBase, MPI_SUM, rhs.father->getMPI());
            res = res.array().pow(1.0 / P).matrix();
            if (average)
                res *= 1. / (rescBase + verySmallReal);
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
                        resc = resc.array().max(uR.array().abs());
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
    void EulerEvaluator<model>::LimiterUGrad(ArrayDOFV<nVarsFixed> &u, ArrayGRADV<nVarsFixed, gDim> &uGrad, ArrayGRADV<nVarsFixed, gDim> &uGradNew)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        static const real safetyRatio = 1 - 1e-5;
        static const real E_lb_eps = smallReal;

        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            uGradNew[iCell] = uGrad[iCell];
            auto c2f = mesh->cell2face[iCell];

            TU_Batch uFaceInc;
            uFaceInc.resize(nVars, c2f.size());
            TU uOtherMin = u[iCell];
            TU uOtherMax = u[iCell];
            auto fEInternal = [](const TU &u) -> real
            { return u(I4) - 0.5 * u(Seq123).squaredNorm() / (u(0) + verySmallReal); };
            real eOtherMin, eOtherMax;
            eOtherMin = eOtherMax = fEInternal(u[iCell]);
            for (rowsize ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                uFaceInc(Eigen::all, ic2f) =
                    uGrad[iCell].transpose() *
                    (vfv->GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, -1) - vfv->GetCellQuadraturePPhys(iCell, -1))(SeqG012);
                index iCellOther = mesh->CellFaceOther(iCell, iFace);
                if (iCellOther != UnInitIndex)
                {
                    uOtherMin = uOtherMin.array().min(u[iCellOther].array());
                    uOtherMax = uOtherMin.array().max(u[iCellOther].array());
                    eOtherMin = std::min(eOtherMin, fEInternal(u[iCellOther]));
                    eOtherMax = std::max(eOtherMax, fEInternal(u[iCellOther]));
                }
            }

            TU uFaceIncMax = uFaceInc.array().rowwise().maxCoeff();
            TU uFaceIncMin = uFaceInc.array().rowwise().minCoeff();
            TU alpha0;
            alpha0.setConstant(nVars, 1.0);
            alpha0 = alpha0.array().min(((uOtherMax - u[iCell]).array().abs() / (uFaceIncMax.array().abs() + verySmallReal)));
            alpha0 = alpha0.array().min(((uOtherMin - u[iCell]).array().abs() / (uFaceIncMin.array().abs() + verySmallReal)));

            uGradNew[iCell].array().rowwise() *= alpha0.array().transpose();
            uFaceInc.array().colwise() *= alpha0.array();

            // start PP

            TU_Batch uFaceAlpha0 = uFaceInc.colwise() + u[iCell];
            real minEFace =
                (uFaceAlpha0(I4, Eigen::all).array() -
                 0.5 * uFaceAlpha0(Seq123, Eigen::all).colwise().squaredNorm().array() / (uFaceAlpha0(0, Eigen::all).array() + verySmallReal))
                    .minCoeff();
            real eC = fEInternal(u[iCell]);
            real deltaEFaceMin = minEFace - eC;
            real alphaPP_E = 1.0;
            if (deltaEFaceMin < 0)
                alphaPP_E = std::min(alphaPP_E, std::abs(eC * (1 - E_lb_eps)) / (verySmallReal - deltaEFaceMin));
            if (alphaPP_E < 1.0)
                alphaPP_E *= safetyRatio;
            uGradNew[iCell](Eigen::all, Seq01234) *= alphaPP_E;
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
        static const real safetyRatio = 1 - 1e-5;
        static const real minRatio = 0.5;
        real rhoEps = smallReal * settings.refUPrim(0) * 1e-1;
        real pEps = smallReal * settings.refUPrim(I4) * 1e-1;
        real betaCutOff = 1e-3;
        bool restrictOnVolPoints = (!settings.ignoreSourceTerm) || settings.forceVolURecBeta;

        if (settings.ppEpsIsRelaxed)
        {
            real rhoMin = veryLargeReal;
            real pMin = veryLargeReal;
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                TU UPrim;
                Gas::IdealGasThermalConservative2Primitive(u[iCell], UPrim, settings.idealGasProperty.gamma);
                rhoMin = std::min(rhoMin, UPrim(0));
                pMin = std::min(pMin, UPrim(I4));
            }
            MPI::AllreduceOneReal(rhoMin, MPI_MIN, mesh->getMPI());
            MPI::AllreduceOneReal(pMin, MPI_MIN, mesh->getMPI());
            rhoEps = std::min(rhoEps, minRatio * rhoMin);
            pEps = std::min(pEps, minRatio * pMin);
        }

        index nLimLocal = 0;
        real minBetaLocal = 1;
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            auto gCell = vfv->GetCellQuad(iCell);
            int nPoint = restrictOnVolPoints ? gCell.GetNumPoints() : 0;
            auto c2f = mesh->cell2face[iCell];
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                nPoint += vfv->GetFaceQuad(c2f[ic2f]).GetNumPoints();
            /***********/
            MatrixXR quadBase;
            quadBase.resize(nPoint, vfv->GetCellAtr(iCell).NDOF - 1);
            nPoint = 0;
            if (restrictOnVolPoints)
            {
                for (int iG = 0; iG < gCell.GetNumPoints(); iG++)
                    quadBase(iG, Eigen::all) = vfv->GetIntPointDiffBaseValue(iCell, -1, -1, iG, 0, 1);
                nPoint += gCell.GetNumPoints();
            }
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                auto gFace = vfv->GetFaceQuad(c2f[ic2f]);
                for (int iG = 0; iG < gFace.GetNumPoints(); iG++)
                    quadBase(nPoint + iG, Eigen::all) = vfv->GetIntPointDiffBaseValue(iCell, c2f[ic2f], -1, iG, 0, 1);
                nPoint += gFace.GetNumPoints();
            }
            /***********/
            DNDS_assert_info(u[iCell](0) >= rhoEps, fmt::format("rhoMean {}, {}", u[iCell](0), rhoEps));
            real gamma = settings.idealGasProperty.gamma;
            real pCent = (gamma - 1) * (u[iCell](I4) - 0.5 * u[iCell](Seq123).squaredNorm() / u[iCell](0));
            DNDS_assert_info(pCent >= pEps, fmt::format("pMean {}, {}", pCent, pEps));

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
                continue; //! early exit, reconstruction is good it self
            }
            if (flag & EvaluateURecBeta_COMPRESS_TO_MEAN)
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
                real thetaC = 1;
                static real v1Eps = smallReal * settings.refUPrim(I4 + 1);
                static real v2Eps = smallReal * settings.refUPrim(I4 + 2);
                Eigen::Vector<real, Eigen::Dynamic> v1S = recInc(Eigen::all, I4 + 1) + recBase(Eigen::all, I4 + 1);
                Eigen::Vector<real, Eigen::Dynamic> v2S = recInc(Eigen::all, I4 + 2) + recBase(Eigen::all, I4 + 2);
                real v1Min = v1S.minCoeff();
                real v2Min = v2S.minCoeff();
                if (v1Min < v1Eps)
                    for (int iG = 0; iG < rhoS.size(); iG++)
                        if (recInc(iG, I4 + 1) < 0) // negative increment
                            thetaC = std::min(thetaC, (recBase(iG, I4 + 1) - v1Eps) / (-recInc(iG, I4 + 1) + verySmallReal));
                if (v2Min < v2Eps)
                    for (int iG = 0; iG < rhoS.size(); iG++)
                        if (recInc(iG, I4 + 2) < 0) // negative increment
                            thetaC = std::min(thetaC, (recBase(iG, I4 + 2) - v2Eps) / (-recInc(iG, I4 + 2) + verySmallReal));
                // theta1 = std::min(theta1, thetaC);
                if (thetaC < 1) // 2eq's pp not disturbing main flow
                {
                    uRec[iCell](Eigen::all, {I4 + 1, I4 + 2}) *= thetaC * 0.9;
                }
            }

            if constexpr (model == NS_SA or model == NS_SA_3D or model == NS_2EQ or model == NS_2EQ_3D)
                recInc(Eigen::all, Seq01234) *= theta1; // to leave SA unchanged
            else
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
                uRecBeta[iCell](0) *= std::pow(uRecBeta[iCell](0), static_cast<int>(std::round(settings.uRecBetaCompressPower))) * safetyRatio;
                minBetaLocal = std::min(uRecBeta[iCell](0), minBetaLocal);
                if (uRecBeta[iCell](0) < betaCutOff)
                    uRecBeta[iCell](0) = 0;
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
        real rhoEps = smallReal * settings.refUPrim(0) * 1e-1;
        real pEps = smallReal * settings.refUPrim(I4) * 1e-1;
        if (settings.ppEpsIsRelaxed)
            rhoEps *= 0, pEps *= 0;

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
        ArrayDOFV<1> &cellRHSAlpha, index &nLim, real &alphaMin,
        real relax, int compress,
        int flag)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        real rhoEps = smallReal * settings.refUPrim(0) * 1e-1;
        real pEps = smallReal * settings.refUPrim(I4) * 1e-1;
        static const real safetyRatio = 1 - 1e-5;
        static const real minRatio = 0.5;

        if (settings.ppEpsIsRelaxed)
        {
            pEps *= 0, rhoEps *= 0;
            DNDS_assert_info(relax < 1, "Relaxed eps only for using relaxation in alpha");
        }

        index nLimLocal = 0;
        real alphaMinLocal = 1;
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            real gamma = settings.idealGasProperty.gamma;
            real alphaRho = 1;
            TU inc = res[iCell];
            DNDS_assert(u[iCell](0) >= rhoEps);
            real relaxedRho = rhoEps * relax + (u[iCell](0)) * (1 - relax);
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
                // static real v1Eps = smallReal * settings.refUPrim(I4 + 1);
                // static real v2Eps = smallReal * settings.refUPrim(I4 + 2);
                // if (inc(I4 + 1) < 0)
                //     alphaRho = std::min(alphaRho,
                //                         (u[iCell](I4 + 1) - v1Eps) / (-inc(I4 + 1) - smallReal * inc(I4 + 1)));
                // if (inc(I4 + 2) < 0)
                //     alphaRho = std::min(alphaRho,
                //                         (u[iCell](I4 + 2) - v2Eps) / (-inc(I4 + 2) - smallReal * inc(I4 + 2)));
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
                cellRHSAlpha[iCell](0) = std::pow(cellRHSAlpha[iCell](0), compress * static_cast<int>(std::round(settings.uRecAlphaCompressPower)));
                nLimLocal++,
                    cellRHSAlpha[iCell] *= safetyRatio,
                    alphaMinLocal = std::min(alphaMinLocal, cellRHSAlpha[iCell](0));
            } //! for safety
        }
        MPI::Allreduce(&nLimLocal, &nLim, 1, DNDS_MPI_INDEX, MPI_SUM, u.father->getMPI().comm);
        MPI::Allreduce(&alphaMinLocal, &alphaMin, 1, DNDS_MPI_REAL, MPI_MIN, u.father->getMPI().comm);
        if (flag & EvaluateCellRHSAlpha_MIN_IF_NOT_ONE)
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                if (cellRHSAlpha[iCell](0) < 1)
                    cellRHSAlpha[iCell](0) = alphaMin;
            }
        if (flag & EvaluateCellRHSAlpha_MIN_ALL)
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
        real rhoEps = smallReal * settings.refUPrim(0) * 1e-1;
        real pEps = smallReal * settings.refUPrim(I4) * 1e-1;

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
                // static real v1Eps = smallReal * settings.refUPrim(I4 + 1);
                // static real v2Eps = smallReal * settings.refUPrim(I4 + 2);
                // if (uNew(I4 + 1) < v1Eps)
                //     cellRHSAlpha[iCell](0) = alphaMin;
                // if (uNew(I4 + 2) < v2Eps)
                //     cellRHSAlpha[iCell](0) = alphaMin;
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

    template <EulerModel model>
    void EulerEvaluator<model>::updateBCProfiles(ArrayDOFV<nVarsFixed> &u, ArrayRECV<nVarsFixed> &uRec)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        for (Geom::t_index i = Geom::BC_ID_DEFAULT_MAX; i < pBCHandler->size(); i++) // init code, consider adding to ctor
        {
            if (pBCHandler->GetFlagFromIDSoft(i, "anchorOpt") != 2)
                continue;
            if (!profileRecorders.count(i))
            {
                real RMin = veryLargeReal;
                real RMax = -veryLargeReal;
                profileRecorders.emplace(std::make_pair(i, OneDimProfile<nVarsFixed>(mesh->getMPI())));
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
                    if (faceBndID == i)
                    {
                        Geom::tSmallCoords coo;
                        mesh->GetCoordsOnFace(iFace, coo);
                        for (int ic = 0; ic < coo.cols(); ic++)
                        {
                            real r = settings.frameConstRotation.rVec(coo(Eigen::all, ic)).norm();
                            RMin = std::min(r, RMin);
                            RMax = std::max(r, RMax);
                        }
                    }
                }
                MPI::AllreduceOneReal(RMin, MPI_MIN, mesh->getMPI());
                MPI::AllreduceOneReal(RMax, MPI_MAX, mesh->getMPI());
                auto vExtra = pBCHandler->GetValueExtraFromID(i);
                // * valueExtra[0] == nDiv
                // * valueExtra[1] == divMethod
                // * valueExtra[2] == d0
                // * valueExtra[3] == printInfo
                index nDiv = vExtra.size() >= 1 ? vExtra(0) : 10;
                index divMethod = vExtra.size() >= 2 ? vExtra(1) : 0; // TODO: implement other distributions
                real divd0 = vExtra.size() >= 3 ? vExtra(2) : veryLargeReal;

                if (divMethod == 0)
                    profileRecorders.at(i).GenerateUniform(std::max(nDiv, index(10)), nVars, RMin, RMax);
                else
                    profileRecorders.at(i).GenerateTanh(std::max(nDiv, index(10)), nVars, RMin, RMax, divd0);
            }
        }
        for (auto &v : profileRecorders)
            v.second.SetZero();
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

            if (pBCHandler->GetFlagFromIDSoft(faceBndID, "anchorOpt") != 2)
                continue;

            real RMin = veryLargeReal;
            real RMax = -veryLargeReal;

            Geom::tSmallCoords coo;
            mesh->GetCoordsOnFace(iFace, coo);
            for (int ic = 0; ic < coo.cols(); ic++)
            {
                real r = settings.frameConstRotation.rVec(coo(Eigen::all, ic)).norm();
                RMin = std::min(r, RMin);
                RMax = std::max(r, RMax);
            }
            TU valIn = u[f2c[0]];
            valIn(Seq123) = settings.frameConstRotation.rtzFrame(vfv->GetFaceQuadraturePPhys(iFace, -1)).transpose()(Seq012, Seq012) * valIn(Seq123); // to rtz frame
#ifndef USE_ABS_VELO_IN_ROTATION
            valIn(2) += valIn(0) * settings.frameConstRotation.Omega() * settings.frameConstRotation.rVec(vfv->GetFaceQuadraturePPhys(iFace, -1)).norm(); // to static value
#endif
            // std::cout << valIn.transpose() << std::endl;
            // std::cout << RMin << " " << RMax << " " << vfv->GetFaceArea(iFace) << std::endl;
            profileRecorders.at(faceBndID).AddSimpleInterval(valIn, vfv->GetFaceArea(iFace), RMin, RMax);
        }
        for (auto &v : profileRecorders)
            v.second.Reduce();
    }

    template <EulerModel model>
    void EulerEvaluator<model>::updateBCProfilesPressureRadialEq()
    {
        for (auto &v : profileRecorders)
        {
            if (pBCHandler->GetFlagFromIDSoft(v.first, "anchorOpt") == 2)
            {
                v.second.v.array().rowwise() /= (v.second.div.array() + verySmallReal);
                v.second.div.setConstant(1.);
                v.second.v(I4, 0) = 0;
                for (index i = 1; i < v.second.Size(); i++)
                {
                    real vt0 = v.second.v(2, i - 1) / v.second.v(0, i - 1);
                    real vt1 = v.second.v(2, i) / v.second.v(0, i);
                    real l0 = v.second.Len(i - 1);
                    real l1 = v.second.Len(i);
                    real ldist = 0.5 * (l0 + l1);
                    real vtm = (vt0 * l0 + vt1 * l1) / (l0 + l1);
                    real rhom = (v.second.v(0, i - 1) * l0 + v.second.v(0, i) * l1) / (l0 + l1);
                    real rc = v.second.nodes[i];
                    v.second.v(I4, i) = v.second.v(I4, i - 1) + rhom * sqr(vtm) / rc * ldist;
                }
                if (mesh->getMPI().rank == 0)
                {
                    // std::cout << "nodes";
                    // for (auto vv : v.second.nodes)
                    //     std::cout << vv << " ";
                    // std::cout << "\n";
                    auto vExtra = pBCHandler->GetValueExtraFromID(v.first);
                    int showMethod = vExtra.size() >= 4 ? vExtra(3) : 0;
                    if (showMethod)
                        log() << fmt::format("EulerEvaluator<model>::updateBCProfilesPressureRadialEq: p rise: [{:.3e}]", v.second.v(I4, Eigen::last))
                              << std::endl;
                }
            }
        }
    }
}
