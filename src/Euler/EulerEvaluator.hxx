/** @file EulerEvaluator.hxx
 *  @brief Template implementations of EulerEvaluator methods for implicit time-stepping,
 *         LU-SGS preconditioning, DOF initialization, boundary value generation,
 *         reconstruction limiting, norm evaluation, and BC profile updates.
 */
#pragma once

#include "DNDS/Defines.hpp" // for correct  DNDS_SWITCH_INTELLISENSE
#include "EulerEvaluator.hpp"
#include "DNDS/HardEigen.hpp"
#include <sstream>
#include <iomanip>
#include "SpecialFields.hpp"
#include "DNDS/ExprtkWrapper.hpp"

namespace DNDS::Euler
{

    static const auto model = NS_SA; // to be hidden by template params

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>, )
    /** @brief Assemble diagonal Jacobian block for each cell for LU-SGS preconditioning.
     *
     *  Computes the spectral radius contribution from face eigenvalues and accumulates
     *  into JDiag. When settings.useRoeJacobian is enabled, also computes Roe flux
     *  Jacobian block contributions including boundary face linearization.
     *
     *  @param JDiag      Diagonal Jacobian block storage (output, cleared then filled).
     *  @param JSource    Source-term Jacobian diagonal block (must match JDiag block mode).
     *  @param dTau       Local pseudo-time step per cell.
     *  @param dt         Global physical time step.
     *  @param alphaDiag  Diagonal scaling factor for the spectral radius.
     *  @param u          Conservative variable DOF array.
     *  @param uRec       Reconstruction coefficients.
     *  @param jacobianCode  Jacobian assembly mode (must be 0).
     *  @param t          Current simulation time.
     */
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
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            auto c2f = mesh->cell2face[iCell];

            // LUSGS diag part
            real fpDivisor = 1.0 / dTau[iCell](0) + 1.0 / dt;
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto if2c = mesh->CellIsFaceBack(iCell, iFace, ic2f) ? 0 : 1;
                fpDivisor += (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) * lambdaFace[iFace] / vfv->GetCellVol(iCell);
                if (!settings.useRoeJacobian)
                    continue;
                // roe term jacobi
                index iCellOther = mesh->CellFaceOther(iCell, iFace, ic2f);
                TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCell, if2c, -1)(Seq012) *
                                (if2c ? -1 : 1); // faces out
                TU uj;
                if (iCellOther == UnInitIndex) // handle BC
                {
                    TU UL = u[iCell];
                    uj = this->generateBoundaryValue(UL, u[iCell], iCell, iFace, -1,
                                                     unitNorm,
                                                     Geom::NormBuildLocalBaseV<dim>(unitNorm),
                                                     vfv->GetFaceQuadraturePPhys(iFace, -1),
                                                     t,
                                                     mesh->GetFaceZone(iFace),
                                                     false, 0);
                }
                else
                    uj = u[iCellOther];
                if (iCellOther != UnInitIndex)
                    this->UFromOtherCell(uj, iFace, iCell, iCellOther, if2c);
                TJacobianU jacII = fluxJacobian0_Right_Times_du_AsMatrix( // unitnorm and uj are both respect with this cell
                    u[iCell], uj,
                    unitNorm, GetFaceVGridFromCell(iFace, iCell, if2c, -1),
                    mesh->GetFaceZone(iFace),
                    lambdaFace[iFace], lambdaFaceC[iFace], lambdaFaceVis[iFace],
                    if2c ? lambdaFace4[iFace] : lambdaFace0[iFace], lambdaFace123[iFace], if2c ? lambdaFace0[iFace] : lambdaFace4[iFace],
                    // swap lambda0 and lambda4 if if2c==1
                    true, +1, 1);              // for this is diff(uthis) not diff(uthat)
                if (iCellOther == UnInitIndex) // handle BC
                {
                    TJacobianU JBC;
                    JBC.resize(nVars, nVars);
                    JBC.setIdentity();
                    for (int i = 0; i < nVars; i++)
                    {
                        TU VE = JBC(EigenAll, i);
                        JBC(EigenAll, i) = this->generateBoundaryValue(VE, u[iCell], iCell, iFace, -1,
                                                                       unitNorm,
                                                                       Geom::NormBuildLocalBaseV<dim>(unitNorm),
                                                                       vfv->GetFaceQuadraturePPhys(iFace, -1),
                                                                       t,
                                                                       mesh->GetFaceZone(iFace),
                                                                       false, 0, /*linMode=*/1);
                    }
                    TJacobianU jacIJ = fluxJacobian0_Right_Times_du_AsMatrix( // unitnorm and uj are both respect with this cell
                        uj, u[iCell],
                        unitNorm, GetFaceVGridFromCell(iFace, iCell, if2c, -1),
                        mesh->GetFaceZone(iFace),
                        lambdaFace[iFace], lambdaFaceC[iFace], lambdaFaceVis[iFace],
                        if2c ? lambdaFace4[iFace] : lambdaFace0[iFace], lambdaFace123[iFace], if2c ? lambdaFace0[iFace] : lambdaFace4[iFace],
                        // swap lambda0 and lambda4 if if2c==1
                        true, -1, 0);
                    JDiag.getBlock(iCell) += (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) * (jacIJ * JBC);
                }
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
                {
                    auto js = JSource.getBlock(iCell);
                    if (!js.allFinite())
                        fprintf(stderr, "[JSource] cell=%ld hasNonFinite\n", long(iCell));
                    JDiag.getBlock(iCell) += alphaDiag * js;
                }
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

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>, )
    /** @brief Compute the implicit matrix-vector product A*uInc for GMRES.
     *
     *  Evaluates the action of the implicit operator (diagonal block + off-diagonal
     *  flux Jacobian contributions) on the increment vector, storing the result in AuInc.
     *
     *  @param alphaDiag  Diagonal scaling factor.
     *  @param t          Current simulation time.
     *  @param u          Conservative variable DOF array.
     *  @param uInc       Increment vector to multiply.
     *  @param JDiag      Diagonal Jacobian block.
     *  @param AuInc      Result of the matrix-vector product (output).
     */
    void EulerEvaluator<model>::LUSGSMatrixVec(
        real alphaDiag,
        real t,
        ArrayDOFV<nVarsFixed> &u,
        ArrayDOFV<nVarsFixed> &uInc,
        JacobianDiagBlock<nVarsFixed> &JDiag,
        ArrayDOFV<nVarsFixed> &AuInc)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "LUSGSMatrixVec 1");
        int cnvars = nVars;

        auto cellOp = [&](index iCell) __attribute__((always_inline)){

        };

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (int iPart = 0; iPart < mesh->NLocalParts(); iPart++)
            for (index iScan = mesh->LocalPartStart(iPart); iScan < mesh->LocalPartEnd(iPart); iScan++)
            {
                index iCell = iScan;
                cellOp(iCell);
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
                    // A self-periodic face can have face2cell = (iCell, iCell),
                    // so face2cell alone cannot identify the side. Use the
                    // cell-local incidence slot to recover if2c before forming
                    // geometry, periodic transforms, and the diagonal LU block.
                    index iCellOther = mesh->CellFaceOther(iCell, iFace, ic2f);
                    auto if2c = mesh->CellIsFaceBack(iCell, iFace, ic2f) ? 0 : 1;
                    TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCell, if2c, -1)(Seq012) *
                                    (if2c ? -1 : 1); // faces out
                    if (iCellOther != UnInitIndex)
                    {

                        if (true)
                        {
                            TU uINCj = uInc[iCellOther];
                            TU uj = u[iCellOther];
                            this->UFromOtherCell(uINCj, iFace, iCell, iCellOther, if2c);
                            this->UFromOtherCell(uj, iFace, iCell, iCellOther, if2c);
                            TU fInc;
                            {

                                // fInc = fluxJacobian0_Right(
                                //            u[iCellOther],
                                //            unitNorm,
                                //            BoundaryType::Inner) *
                                //        uInc[iCellOther]; //! always inner here
                                fInc = fluxJacobian0_Right_Times_du(
                                    uj, u[iCell],
                                    unitNorm, GetFaceVGridFromCell(iFace, iCell, if2c, -1),
                                    Geom::BC_ID_INTERNAL, uINCj,
                                    lambdaFace[iFace], lambdaFaceC[iFace], lambdaFaceVis[iFace],
                                    if2c ? lambdaFace4[iFace] : lambdaFace0[iFace], lambdaFace123[iFace], if2c ? lambdaFace0[iFace] : lambdaFace4[iFace],
                                    // swap lambda0 and lambda4 if if2c==1
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
                AuInc[iCell] = JDiag.MatVecLeft(iCell, uInc[iCell]) - uIncNewBuf;

                auto AuIncI = AuInc[iCell];
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
    /** @brief Assemble and factorize the local LU Jacobian from diagonal and off-diagonal blocks.
     *
     *  Populates the sparse local LU structure with diagonal blocks from JDiag and
     *  off-diagonal Roe flux Jacobian entries for local-partition neighbor cells,
     *  then performs in-place LU decomposition.
     *
     *  @param alphaDiag  Diagonal scaling factor for face flux Jacobian.
     *  @param t          Current simulation time.
     *  @param u          Conservative variable DOF array.
     *  @param JDiag      Diagonal Jacobian block.
     *  @param jacLU      Local LU factorization structure (output).
     */
    void EulerEvaluator<model>::LUSGSMatrixToJacobianLU(
        real alphaDiag, real t,
        ArrayDOFV<nVarsFixed> &u,
        JacobianDiagBlock<nVarsFixed> &JDiag,
        JacobianLocalLU<nVarsFixed> &jacLU)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "LUSGSMatrixToJacobianLU 1");
        int cnvars = nVars;
        jacLU.setZero();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (int iPart = 0; iPart < mesh->NLocalParts(); iPart++)
            for (index iScan = mesh->LocalPartStart(iPart); iScan < mesh->LocalPartEnd(iPart); iScan++)
            {
                index iCell = iScan;
                jacLU.GetDiag(iCell) = JDiag.getValue(iCell);
                auto c2f = mesh->cell2face[iCell];
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    index iCellOther = mesh->CellFaceOther(iCell, iFace, ic2f);
                    auto if2c = mesh->CellIsFaceBack(iCell, iFace, ic2f) ? 0 : 1;
                    TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCell, if2c, -1)(Seq012) *
                                    (if2c ? -1 : 1); // faces out
                    if (iCellOther != UnInitIndex &&
                        iCellOther < mesh->LocalPartEnd(iPart) && iCellOther >= mesh->LocalPartStart(iPart))
                    {
                        TU uj = u[iCellOther];
                        this->UFromOtherCell(uj, iFace, iCell, iCellOther, if2c);
                        TJacobianU jacIJ;
                        {
                            jacIJ = fluxJacobian0_Right_Times_du_AsMatrix( // unitnorm and uj are both respect with this cell
                                uj, u[iCell],
                                unitNorm, GetFaceVGridFromCell(iFace, iCell, if2c, -1),
                                Geom::BC_ID_INTERNAL,
                                lambdaFace[iFace], lambdaFaceC[iFace], lambdaFaceVis[iFace],
                                if2c ? lambdaFace4[iFace] : lambdaFace0[iFace], lambdaFace123[iFace], if2c ? lambdaFace0[iFace] : lambdaFace4[iFace],
                                // swap lambda0 and lambda4 if if2c==1
                                settings.useRoeJacobian); //! always inner here
                        }
                        auto faceID = mesh->GetFaceZone(iFace);
                        mesh->CellOtherCellPeriodicHandle(
                            iFace, if2c,
                            [&]()
                            { jacIJ(EigenAll, Seq123) =
                                  mesh->periodicInfo.TransVectorBack<dim, nVarsFixed>(
                                                        jacIJ(EigenAll, Seq123).transpose(), faceID)
                                      .transpose(); },
                            [&]()
                            { jacIJ(EigenAll, Seq123) =
                                  mesh->periodicInfo.TransVector<dim, nVarsFixed>(
                                                        jacIJ(EigenAll, Seq123).transpose(), faceID)
                                      .transpose(); });
                        auto jacBlock = (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) * jacIJ;
                        if (iCellOther == iCell)
                        {
                            jacLU.GetDiag(iCell) += jacBlock;
                        }
                        else
                        {
                            int iC2CInLocal = -1;
                            for (int ic2c = 0; ic2c < mesh->cell2cellFaceVLocalParts[iCell].size(); ic2c++)
                                if (iCellOther == mesh->cell2cellFaceVLocalParts[iCell][ic2c])
                                    iC2CInLocal = ic2c; // TODO: pre-search this
                            DNDS_assert(iC2CInLocal != -1);
                            jacLU.LDU(iCell, symLU->cell2cellFaceVLocal2FullRowPos[iCell][iC2CInLocal]) += jacBlock;
                        }
                    }
                }
            }
        // TODO: make below OMP-ed
        jacLU.InPlaceDecompose();
        DNDS_MPI_InsertCheck(u.father->getMPI(), "LUSGSMatrixToJacobianLU -1");
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>, template <>)
    /** @brief [[deprecated]] Forward LU-SGS sweep. Use UpdateSGS with uIncIsZero=true instead.
     *
     *  Performs a forward Gauss-Seidel sweep using diagonal inversion and lower-triangular
     *  off-diagonal flux Jacobian contributions. Retained for callers that alias uInc==uIncNew.
     *
     *  @param alphaDiag  Diagonal scaling factor.
     *  @param t          Current simulation time.
     *  @param rhs        Right-hand side residual.
     *  @param u          Conservative variable DOF array.
     *  @param uInc       Current increment (input, may alias uIncNew).
     *  @param JDiag      Diagonal Jacobian block.
     *  @param uIncNew    Updated increment (output, may alias uInc).
     */
    void EulerEvaluator<model>::UpdateLUSGSForward(
        real alphaDiag, real t,
        ArrayDOFV<nVarsFixed> &rhs,
        ArrayDOFV<nVarsFixed> &u,
        ArrayDOFV<nVarsFixed> &uInc,
        JacobianDiagBlock<nVarsFixed> &JDiag,
        ArrayDOFV<nVarsFixed> &uIncNew)
    {
        // Deprecated: delegate to UpdateSGS with uIncIsZero.
        // Note: callers pass uInc == uIncNew (aliased), but UpdateSGS requires
        // uInc != uIncNew. We keep the original implementation for now to
        // preserve this aliasing behavior. Migrate callers to use UpdateSGS
        // with separate buffers instead.
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateLUSGSForward 1");
        int cnvars = nVars;
        JDiag.GetInvert();
        index nCellDist = mesh->NumCell();
        for (index iScan = 0; iScan < nCellDist; iScan++)
        {
            index iCell = iScan;

            auto c2f = mesh->cell2face[iCell];
            TU uIncNewBuf(nVars);
            auto RHSI = rhs[iCell];
            uIncNewBuf = RHSI;

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                index iCellOther = mesh->CellFaceOther(iCell, iFace, ic2f);
                auto if2c = mesh->CellIsFaceBack(iCell, iFace, ic2f) ? 0 : 1;
                TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCell, if2c, -1)(Seq012) *
                                (if2c ? -1 : 1); // faces out
                if (iCellOther != UnInitIndex)
                {
                    index iScanOther = iCellOther;
                    if (iScanOther < iScan)
                    {
                        TU uINCj = uInc[iCellOther];
                        TU uj = u[iCellOther];
                        this->UFromOtherCell(uINCj, iFace, iCell, iCellOther, if2c);
                        this->UFromOtherCell(uj, iFace, iCell, iCellOther, if2c);

                        TU fInc = fluxJacobian0_Right_Times_du(
                            uj, u[iCell],
                            unitNorm, GetFaceVGridFromCell(iFace, iCell, if2c, -1),
                            Geom::BC_ID_INTERNAL, uINCj,
                            lambdaFace[iFace], lambdaFaceC[iFace], lambdaFaceVis[iFace],
                            if2c ? lambdaFace4[iFace] : lambdaFace0[iFace], lambdaFace123[iFace], if2c ? lambdaFace0[iFace] : lambdaFace4[iFace],
                            settings.useRoeJacobian);

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
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>, template <>)
    /** @brief [[deprecated]] Backward LU-SGS sweep. Use UpdateSGS instead.
     *
     *  Performs a backward Gauss-Seidel sweep using diagonal inversion and upper-triangular
     *  off-diagonal flux Jacobian contributions. Adds the correction to uIncNew in-place.
     *
     *  @param alphaDiag  Diagonal scaling factor.
     *  @param t          Current simulation time.
     *  @param rhs        Right-hand side residual.
     *  @param u          Conservative variable DOF array.
     *  @param uInc       Current increment (input).
     *  @param JDiag      Diagonal Jacobian block.
     *  @param uIncNew    Updated increment (input/output, correction added).
     */
    void EulerEvaluator<model>::UpdateLUSGSBackward(
        real alphaDiag, real t,
        ArrayDOFV<nVarsFixed> &rhs,
        ArrayDOFV<nVarsFixed> &u,
        ArrayDOFV<nVarsFixed> &uInc,
        JacobianDiagBlock<nVarsFixed> &JDiag,
        ArrayDOFV<nVarsFixed> &uIncNew)
    {
        // Deprecated: see UpdateLUSGSForward comment.
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateLUSGSBackward 1");
        int cnvars = nVars;
        JDiag.GetInvert();
        index nCellDist = mesh->NumCell();
        for (index iScan = nCellDist - 1; iScan >= 0; iScan--)
        {
            index iCell = iScan;

            auto c2f = mesh->cell2face[iCell];
            TU uIncNewBuf(cnvars);
            uIncNewBuf.setZero();

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                index iCellOther = mesh->CellFaceOther(iCell, iFace, ic2f);
                auto if2c = mesh->CellIsFaceBack(iCell, iFace, ic2f) ? 0 : 1;
                TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCell, if2c, -1)(Seq012) *
                                (if2c ? -1 : 1); // faces out
                if (iCellOther != UnInitIndex)
                {
                    index iScanOther = iCellOther;
                    if (iScanOther > iScan)
                    {
                        TU uINCj = uInc[iCellOther];
                        TU uj = u[iCellOther];
                        this->UFromOtherCell(uINCj, iFace, iCell, iCellOther, if2c);
                        this->UFromOtherCell(uj, iFace, iCell, iCellOther, if2c);

                        TU fInc = fluxJacobian0_Right_Times_du(
                            uj, u[iCell],
                            unitNorm, GetFaceVGridFromCell(iFace, iCell, if2c, -1),
                            Geom::BC_ID_INTERNAL, uINCj,
                            lambdaFace[iFace], lambdaFaceC[iFace], lambdaFaceVis[iFace],
                            if2c ? lambdaFace4[iFace] : lambdaFace0[iFace], lambdaFace123[iFace], if2c ? lambdaFace0[iFace] : lambdaFace4[iFace],
                            settings.useRoeJacobian);

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
    /** @brief Symmetric Gauss-Seidel (SGS) sweep for implicit preconditioning.
     *
     *  Performs a single forward or backward sweep. When uIncIsZero is true, skips
     *  off-diagonal terms involving the zeroed increment in the first forward pass
     *  to save computation. Accumulates the global L1 increment change into sumInc.
     *
     *  @param alphaDiag   Diagonal scaling factor.
     *  @param t           Current simulation time.
     *  @param rhs         Right-hand side residual.
     *  @param u           Conservative variable DOF array.
     *  @param uInc        Current increment (input, must not alias uIncNew).
     *  @param uIncNew     Updated increment (output).
     *  @param JDiag       Diagonal Jacobian block.
     *  @param forward     If true, sweep from cell 0 to N-1; otherwise N-1 to 0.
     *  @param gsUpdate    If true, use Gauss-Seidel update (use latest values); otherwise Jacobi.
     *  @param sumInc      Global L1 sum of increment changes (output, MPI-reduced).
     *  @param uIncIsZero  If true, skip off-diagonal terms for zero-initialized increment.
     */
    void EulerEvaluator<model>::UpdateSGS(
        real alphaDiag, real t,
        ArrayDOFV<nVarsFixed> &rhs,
        ArrayDOFV<nVarsFixed> &u,
        ArrayDOFV<nVarsFixed> &uInc,
        ArrayDOFV<nVarsFixed> &uIncNew,
        JacobianDiagBlock<nVarsFixed> &JDiag,
        bool forward, bool gsUpdate, TU &sumInc,
        bool uIncIsZero)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateSGS 1");
        DNDS_assert(&uInc != &uIncNew);
        int cnvars = nVars;
        JDiag.GetInvert();
        const index nCellDist = mesh->NumCell();
        sumInc.setZero(cnvars);

#if defined(DNDS_DIST_MT_USE_OMP)
        auto cellOp = [&](index iScanStart, index iScanEnd)
        {
#else
        const index iScanStart{0}, iScanEnd(nCellDist);
#endif
            for (index iScan = iScanStart; iScan < iScanEnd; iScan++)
            {
                index iCell = forward ? iScan : iScanEnd - 1 - (iScan - iScanStart);

                auto c2f = mesh->cell2face[iCell];
                TU uIncNewBuf(nVars);
                auto RHSI = rhs[iCell];
                // std::cout << rhs[iCell](0) << std::endl;
                uIncNewBuf = RHSI;

                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    auto btype = mesh->GetFaceZone(iFace);
                    index iCellOther = mesh->CellFaceOther(iCell, iFace, ic2f);
                    auto if2c = mesh->CellIsFaceBack(iCell, iFace, ic2f) ? 0 : 1;
                    TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCell, if2c, -1)(Seq012) *
                                    (if2c ? -1 : 1); // faces out
                    if (iCellOther != UnInitIndex)
                    {
                        if (iCell != iCellOther)
                        {
                            TU fInc;
                            bool iCellOtherIsThisPart = iScanStart <= iCellOther && iCellOther < iScanEnd;
                            bool gsUseNew = gsUpdate && iCellOtherIsThisPart && (forward ? (iCellOther < iCell) : (iCellOther > iCell));

                            // When uInc is known to be zero, skip flux for not-yet-processed
                            // neighbours whose increment is still zero (LUSGS optimisation).
                            if (uIncIsZero && !gsUseNew)
                                continue;

                            TU uINCj = gsUseNew ? uIncNew[iCellOther] : uInc[iCellOther];
                            TU uj = u[iCellOther];
                            this->UFromOtherCell(uINCj, iFace, iCell, iCellOther, if2c);
                            this->UFromOtherCell(uj, iFace, iCell, iCellOther, if2c);

                            {
                                fInc = fluxJacobian0_Right_Times_du(
                                    uj, u[iCell],
                                    unitNorm, GetFaceVGridFromCell(iFace, iCell, if2c, -1),
                                    Geom::BC_ID_INTERNAL, uINCj,
                                    lambdaFace[iFace], lambdaFaceC[iFace], lambdaFaceVis[iFace],
                                    if2c ? lambdaFace4[iFace] : lambdaFace0[iFace], lambdaFace123[iFace], if2c ? lambdaFace0[iFace] : lambdaFace4[iFace],
                                    // swap lambda0 and lambda4 if if2c==1
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
                    // else if (pBCHandler->GetTypeFromID(btype) == BCWall || pBCHandler->GetTypeFromID(btype) == BCWallIsothermal)
                    // {
                    //     TMat normBase = Geom::NormBuildLocalBaseV<dim>(unitNorm);
                    //     Geom::tPoint pPhysics = vfv->GetFaceQuadraturePPhysFromCell(iFace, iCell, if2c, -1);
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
                    //         unitNorm, GetFaceVGridFromCell(iFace, iCell, if2c, -1),
                    //         Geom::BC_ID_INTERNAL, uINCj,
                    //         lambdaFace[iFace], lambdaFaceC[iFace], lambdaFaceVis[iFace],
                    //         if2c ? lambdaFace4[iFace] : lambdaFace0[iFace], lambdaFace123[iFace], if2c ? lambdaFace0[iFace] : lambdaFace4[iFace],
                    //                // swap lambda0 and lambda4 if if2c==1
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
            }
#if defined(DNDS_DIST_MT_USE_OMP)
        };
#endif

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp declare reduction(TUAdd:TU : omp_out += omp_in) initializer(omp_priv = omp_orig)
#    pragma omp parallel for schedule(static) reduction(TUAdd : sumInc)
        for (int iPart = 0; iPart < mesh->NLocalParts(); iPart++)
            cellOp(mesh->LocalPartStart(iPart), mesh->LocalPartEnd(iPart));
#endif

        TU sumIncAll(cnvars);
        // std::abort();
        MPI::Allreduce(sumInc.data(), sumIncAll.data(), sumInc.size(), DNDS_MPI_REAL, MPI_SUM, rhs.father->getMPI().comm);
        sumInc = sumIncAll;
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateSGS -1");
        // exit(-1);
    }

    template <EulerModel model>
    /** @brief SGS sweep with reconstruction-level off-diagonal terms.
     *
     *  Extends UpdateSGS by including reconstruction increment contributions
     *  (uRecInc) in the off-diagonal flux Jacobian product, enabling higher-order
     *  implicit coupling.
     *
     *  @param alphaDiag  Diagonal scaling factor.
     *  @param t          Current simulation time.
     *  @param rhs        Right-hand side residual.
     *  @param u          Conservative variable DOF array.
     *  @param uRec       Reconstruction coefficients.
     *  @param uInc       Current DOF increment.
     *  @param uRecInc    Current reconstruction increment.
     *  @param JDiag      Diagonal Jacobian block.
     *  @param forward    Sweep direction: true for forward, false for backward.
     *  @param sumInc     Global L1 sum of increment changes (output, MPI-reduced).
     */
    void EulerEvaluator<model>::UpdateSGSWithRec(
        real alphaDiag, real t,
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
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp declare reduction(TUAdd:TU : omp_out += omp_in) initializer(omp_priv = omp_orig)
#    pragma omp parallel for schedule(static) reduction(TUAdd : sumInc)
#endif
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
                index iCellOther = mesh->CellFaceOther(iCell, iFace, ic2f);
                auto if2c = mesh->CellIsFaceBack(iCell, iFace, ic2f) ? 0 : 1;
                TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCell, if2c, -1)(Seq012) *
                                (if2c ? -1 : 1); // faces out
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
                                unitNorm, GetFaceVGridFromCell(iFace, iCell, if2c, -1),
                                Geom::BC_ID_INTERNAL, uINCj,
                                lambdaFace[iFace], lambdaFaceC[iFace], lambdaFaceVis[iFace],
                                if2c ? lambdaFace4[iFace] : lambdaFace0[iFace], lambdaFace123[iFace], if2c ? lambdaFace0[iFace] : lambdaFace4[iFace],
                                // swap lambda0 and lambda4 if if2c==1
                                settings.useRoeJacobian); //! always inner here
                        }
                        {
                            TU uRecSLInc =
                                (vfv->GetIntPointDiffBaseValue(iCell, iFace, if2c, -1, std::array<int, 1>{0}, 1) *
                                 uRecInc[iCell])
                                    .transpose();
                            TU uRecSRInc =
                                (vfv->GetIntPointDiffBaseValue(iCellOther, iFace, 1 - if2c, -1, std::array<int, 1>{0}, 1) *
                                 uRecInc[iCellOther]) //! TODO periodic here
                                    .transpose();
                            TU fIncSL = fluxJacobianC_Right_Times_du(u[iCell], unitNorm, GetFaceVGridFromCell(iFace, iCell, if2c, -1), Geom::BC_ID_INTERNAL, uRecSLInc);
                            TU fIncSR = fluxJacobianC_Right_Times_du(u[iCellOther], unitNorm, GetFaceVGridFromCell(iFace, iCell, if2c, -1), Geom::BC_ID_INTERNAL, uRecSRInc);
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
        template <EulerModel model>, template <>)
    /** @brief Solve the local LU-factorized Jacobian system for the increment.
     *
     *  Applies the pre-factorized local LU solve (from LUSGSMatrixToJacobianLU) to
     *  compute the new increment from the RHS residual and off-diagonal contributions.
     *  Supports accumulation onto a nonzero initial increment when uIncIsZero is false.
     *
     *  @param alphaDiag   Diagonal scaling factor.
     *  @param t           Current simulation time.
     *  @param rhs         Right-hand side residual.
     *  @param u           Conservative variable DOF array.
     *  @param uInc        Current increment (input).
     *  @param uIncNew     Updated increment (output, must not alias uInc).
     *  @param bBuf        Temporary buffer for RHS assembly.
     *  @param JDiag       Diagonal Jacobian block.
     *  @param jacLU       Pre-factorized local LU structure.
     *  @param uIncIsZero  If true, uInc is zero and can be skipped in accumulation.
     *  @param sumInc      Component-wise L1 norm of the new increment (output).
     */
    void EulerEvaluator<model>::LUSGSMatrixSolveJacobianLU(
        real alphaDiag, real t,
        ArrayDOFV<nVarsFixed> &rhs,
        ArrayDOFV<nVarsFixed> &u,
        ArrayDOFV<nVarsFixed> &uInc,
        ArrayDOFV<nVarsFixed> &uIncNew,
        ArrayDOFV<nVarsFixed> &bBuf,
        JacobianDiagBlock<nVarsFixed> &JDiag,
        JacobianLocalLU<nVarsFixed> &jacLU,
        bool uIncIsZero,
        TU &sumInc)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "LUSGSMatrixSolveJacobianLU 1");
        int cnvars = nVars;
        index nCellDist = mesh->NumCell();
        sumInc.setZero(cnvars);
        //- revert to old LU solve here
        if (!uIncIsZero)
            this->LUSGSMatrixVec(alphaDiag, t, u, uInc, JDiag, bBuf); // bBuf = Ax
        else
            bBuf.setConstant(0.0);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (int iPart = 0; iPart < mesh->NLocalParts(); iPart++)
            for (index iScan = mesh->LocalPartStart(iPart); iScan < mesh->LocalPartEnd(iPart); iScan++)
            // update the ghost part (non proc-block) rhs
            {
                index iCell = iScan;
                auto c2f = mesh->cell2face[iCell];
                // std::cout << uIncIsZero << " " << uInc[iCell].norm() << " " << bBuf[iCell] << std::endl;
                bBuf[iCell] = rhs[iCell] - bBuf[iCell]; // bBuf = b - Ax

                // bBuf[iCell] = rhs[iCell]; //+ revert to old LU solve here
                // if (uIncIsZero) //+ revert to old LU solve here
                continue;

                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    index iCellOther = mesh->CellFaceOther(iCell, iFace, ic2f);
                    auto if2c = mesh->CellIsFaceBack(iCell, iFace, ic2f) ? 0 : 1;
                    TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCell, if2c, -1)(Seq012) *
                                    (if2c ? -1 : 1); // faces out
                    if (iCellOther != UnInitIndex && iCell != iCellOther
                        // if is a ghost neighbour
                        && iCellOther >= mesh->NumCell())
                    {
                        TU fInc;
                        TU uINCj = uInc[iCellOther];
                        TU uj = u[iCellOther];
                        this->UFromOtherCell(uINCj, iFace, iCell, iCellOther, if2c);
                        this->UFromOtherCell(uj, iFace, iCell, iCellOther, if2c);
                        {

                            fInc = fluxJacobian0_Right_Times_du(
                                uj, u[iCell],
                                unitNorm, GetFaceVGridFromCell(iFace, iCell, if2c, -1),
                                Geom::BC_ID_INTERNAL, uINCj,
                                lambdaFace[iFace], lambdaFaceC[iFace], lambdaFaceVis[iFace],
                                if2c ? lambdaFace4[iFace] : lambdaFace0[iFace], lambdaFace123[iFace], if2c ? lambdaFace0[iFace] : lambdaFace4[iFace],
                                // swap lambda0 and lambda4 if if2c==1
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
        sumInc = uIncNew.componentWiseNorm1();

        //- revert to old LU solve here
        if (!uIncIsZero)
            uIncNew += uInc;

        DNDS_assert(uIncNew.father.get() != uInc.father.get()); // no aliasing
        // TU sumIncAll(cnvars);
        // MPI::Allreduce(sumInc.data(), sumIncAll.data(), sumInc.size(), DNDS_MPI_REAL, MPI_SUM, rhs.father->getMPI().comm);
        // sumInc = sumIncAll;

        DNDS_MPI_InsertCheck(u.father->getMPI(), "LUSGSMatrixSolveJacobianLU -1");
        // exit(-1);
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>, )
    /** @brief Initialize conservative variable DOF from farfield and special-field initializers.
     *
     *  Sets all cells to the farfield static value, then applies model-specific
     *  initialization (SA wall-distance scaling), box/plane/exprtk region overrides,
     *  and built-in special fields (isentropic vortex, Rayleigh-Taylor, Taylor-Green,
     *  double Mach reflection, etc.) based on settings.
     *
     *  @param u  Conservative variable DOF array (output, fully initialized).
     */
    void EulerEvaluator<model>::InitializeUDOF(ArrayDOFV<nVarsFixed> &u)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        Eigen::VectorXd initConstVal = this->settings.farFieldStaticValue.cons;
        u.setConstant(initConstVal);
        if (model == EulerModel::NS_SA || model == NS_SA_3D)
        {
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (int iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                auto c2f = mesh->cell2face[iCell];
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    if (pBCHandler->GetTypeFromID(mesh->GetFaceZone(iFace)) == EulerBCType::BCWall ||
                        pBCHandler->GetTypeFromID(mesh->GetFaceZone(iFace)) == EulerBCType::BCWallIsothermal)
                        u[iCell](I4 + 1) *= 1.0; // ! not fixing first layer!
                }
            }
        }
        if (model == EulerModel::NS_2EQ || model == NS_2EQ_3D)
        {
            if (settings.ransModel == RANSModel::RANS_KOSST ||
                settings.ransModel == RANSModel::RANS_KOWilcox)
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
                for (int iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    auto c2f = mesh->cell2face[iCell];
                    real d = dWall.at(iCell).mean();
                    // for SST or KOWilcox
                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                    // if (pBCHandler->GetTypeFromID(mesh->GetFaceZone(c2f[ic2f])) == EulerBCType::BCWall ||
                    // pBCHandler->GetTypeFromID(mesh->GetFaceZone(c2f[ic2f])) == EulerBCType::BCWallIsothermal)
                    {
                        auto [T, pMean, asqrMean, Hmean, gammaEq, gamma] = phys_.conservativeThermal(u[iCell]);
                        real muRef = phys_.muRef();
                        real mufPhy1 = muEff(u[iCell], T);
                        real rhoOmegaaaWall = mufPhy1 / sqr(d) * RANS::kWallOmegaCoeff * 0.1;

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
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    Geom::tPoint pos = vfv->GetCellBary(iCell);
                    real T = phys_.temperature(u[iCell]);
                    real gammaEq = phys_.gammaEq(T, u[iCell]);
                    real gamma = phys_.gamma(T, u[iCell]);
                    real rho = 2;
                    real p = 1 + 2 * pos(1);
                    if (pos(1) >= 0.5)
                    {
                        rho = 1;
                        p = 1.5 + pos(1);
                    }
                    real v = -0.025 * sqrt(gamma * p / rho) * std::cos(8 * pi * pos(0));
                    if constexpr (dim == 3)
                        u[iCell] = Eigen::Vector<real, 5>{rho, 0, rho * v, 0, 0.5 * rho * sqr(v) + p / (gammaEq - 1)};
                    else
                        u[iCell] = Eigen::Vector<real, 4>{rho, 0, rho * v, 0.5 * rho * sqr(v) + p / (gammaEq - 1)};
                }
            else if constexpr (model == NS_3D)
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    Geom::tPoint pos = vfv->GetCellBary(iCell);
                    real T = phys_.temperature(u[iCell]);
                    real gammaEq = phys_.gammaEq(T, u[iCell]);
                    real gamma = phys_.gamma(T, u[iCell]);
                    real rho = 2;
                    real p = 1 + 2 * pos(1);
                    if (pos(1) >= 0.5)
                    {
                        rho = 1;
                        p = 1.5 + pos(1);
                    }
                    real v = -0.025 * sqrt(gamma * p / rho) * std::cos(8 * pi * pos(0)) * std::cos(8 * pi * pos(2));
                    u[iCell] = Eigen::Vector<real, 5>{rho, 0, rho * v, 0, 0.5 * rho * sqr(v) + p / (gammaEq - 1)};
                }
            break;
        case 2:   // for IV10 problem
        case 203: // for IV10 problem with PP
        case 202:
            DNDS_assert(model == NS || model == NS_2D || model == NS_EX);
            if constexpr (model == NS || model == NS_2D || model == NS_EX)
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
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
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
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
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    Geom::tPoint pos = vfv->GetCellBary(iCell);
                    real M0 = 0.1;
                    real T_cell = phys_.temperature(u[iCell]);
                    real gammaEq = phys_.gammaEq(T_cell, u[iCell]);
                    real gamma = phys_.gamma(T_cell, u[iCell]);
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
                            real E = 0.5 * (sqr(ux) + sqr(uy)) * rho + p / (gammaEq - 1);

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
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                Geom::tPoint pos = vfv->GetCellBary(iCell);
                real M0 = 0.1;
                real T_far = phys_.temperature(settings.farFieldStaticValue.cons);
                real gamma_far = phys_.gammaEq(T_far, settings.farFieldStaticValue.cons);
                auto c2n = mesh->cell2node[iCell];
                auto gCell = vfv->GetCellQuad(iCell);
                TU um;
                um.resizeLike(u[iCell]);
                um.setZero();
                // Eigen::MatrixXd coords;
                // mesh->GetCoords(c2n, coords);
                gCell.IntegrationSimple(
                    um,
                    [&, gamma_far](TU &inc, int ig)
                    {
                        // std::cout << coords<< std::endl << std::endl;
                        // std::cout << DiNj << std::endl;
                        Geom::tPoint pPhysics = vfv->GetCellQuadraturePPhys(iCell, ig);
                        TU farPrimitive;
                        Gas::IdealGasThermalConservative2Primitive<dim>(settings.farFieldStaticValue.cons, farPrimitive, gamma_far,
                                                                        phys_.mixtureBaseInternalRhoE(settings.farFieldStaticValue.cons));
                        real pInf = farPrimitive(I4);
                        real r = pPhysics.norm();
                        TVec velo = -pPhysics(Seq012) / (r + smallReal);
                        farPrimitive(I4) = pInf;
                        farPrimitive(0) = 1;
                        farPrimitive(Seq123) = velo;

                        Gas::IdealGasThermalPrimitive2Conservative<dim>(farPrimitive, inc, gamma_far, 0 /* config, sensible ρE */);
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
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
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
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                Geom::tPoint pos = vfv->GetCellBary(iCell);
                if (pos(0) > i.x0 && pos(0) < i.x1 &&
                    pos(1) > i.y0 && pos(1) < i.y1 &&
                    pos(2) > i.z0 && pos(2) < i.z1)
                {
                    u[iCell] = i.v.cons;
                }
            }
        }

        // Plane
        for (auto &i : settings.planeInitializers)
        {
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                Geom::tPoint pos = vfv->GetCellBary(iCell);
                if (pos(0) * i.a + pos(1) * i.b + pos(2) * i.c + i.h > 0)
                {
                    // std::cout << pos << std::endl << i.a << i.b << std::endl << i.h <<std::endl;
                    // DNDS_assert(false);
                    u[iCell] = i.v.cons;
                }
            }
        }

        for (auto &i : settings.exprtkInitializers)
        {
            StateValueOrigin stateOrigin = StateValueOriginFromName(i.stateType);
            DNDS_check_throw_info(stateOrigin != StateValueOrigin::None &&
                                      stateOrigin != StateValueOrigin::NonState &&
                                      stateOrigin != StateValueOrigin::Invalid,
                                  fmt::format("unsupported exprtkInitializers stateType [{}]", i.stateType));
            auto exprStr = i.GetExpr();
            auto makeExprtkEvaluator = [&]()
            {
                auto eval = std::make_unique<ExprtkWrapperEvaluator>();
                eval->AddScalar("inRegion");
                eval->AddScalar("iCell");
                eval->AddVector("x", dim);
                eval->AddVector("UExprtk", nVars);
                eval->Compile(exprStr);
                return eval;
            };
            auto exprtkEvalCheck = makeExprtkEvaluator();
            int nExprtkEvaluators = 1;
#if defined(DNDS_DIST_MT_USE_OMP)
            nExprtkEvaluators = omp_get_max_threads();
#endif
            std::vector<std::unique_ptr<ExprtkWrapperEvaluator>> exprtkEvals;
            exprtkEvals.reserve(nExprtkEvaluators);
            for (int iEval = 0; iEval < nExprtkEvaluators; ++iEval)
                exprtkEvals.push_back(makeExprtkEvaluator());
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                int exprtkTid = 0;
#if defined(DNDS_DIST_MT_USE_OMP)
                exprtkTid = omp_get_thread_num();
#endif
                DNDS_check_throw_info(exprtkTid < static_cast<int>(exprtkEvals.size()),
                                      fmt::format("ExprTk evaluator pool has {} entries but OpenMP thread {} requested one",
                                                  exprtkEvals.size(), exprtkTid));
                auto &exprtkEval = *exprtkEvals[exprtkTid];

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
                        uPrimitive.resizeLike(u[iCell]);
                        phys_.conservativeToStateValueOrigin(u[iCell], uPrimitive, stateOrigin);
                        for (int i = 0; i < nVars; i++)
                            exprtkEval.VarVec("UExprtk", i) = uPrimitive(i);

                        real ret = exprtkEval.Evaluate();

                        DNDS_check_throw_info(ret == 0.0, "return of \n" + exprStr + "\nis non-zero");
                        allIn = allIn && exprtkEval.Var("inRegion");
                        someIn = someIn || exprtkEval.Var("inRegion");

                        if (exprtkEval.Var("inRegion"))
                        {
                            for (int i = 0; i < nVars; i++)
                                uPrimitive(i) = exprtkEval.VarVec("UExprtk", i);
                            phys_.stateValueOriginToConservative(uPrimitive, inc, stateOrigin);
                        }
                        else
                        {
                            inc = u[iCell];
                        }
                        if (!inc.allFinite())
                        {
                            std::ostringstream oss0, oss1, oss2;
                            oss0 << uPrimitive.transpose();
                            oss1 << inc.transpose();
                            oss2 << "x=" << pPhysics.transpose();
                            DNDS_check_throw_info(
                                false,
                                fmt::format("Got invalid exprtk product:  [{}] -> [{}]\n{}\n {}\n",
                                            oss0.str(), oss1.str(), oss2.str(), exprStr));
                        }
                        inc *= vfv->GetCellJacobiDet(iCell, ig); // don't forget this
                    });
                // if (allIn)
                if (someIn)
                    u[iCell] = um / vfv->GetCellVol(iCell); // mean value
            }
        }
        // All configured state values and ExprTk products are resolved to
        // conservative-total states before storage.
    }

    template <EulerModel model>
    /** @brief Apply maximum-value spatial filter to conservative variables.
     *
     *  Placeholder for a spatial filter Jacobian approach. Currently returns immediately
     *  without modification.
     *
     *  @param u  Conservative variable DOF array (unmodified).
     */
    void EulerEvaluator<model>::FixUMaxFilter(ArrayDOFV<nVarsFixed> &u)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        // TODO: make spacial filter jacobian
        return; // ! nofix shortcut
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>, )
    void EulerEvaluator<model>::PointImplicitSourceUpdate(
        ArrayDOFV<nVarsFixed> &uNew,
        const ArrayDOFV<nVarsFixed> &res,
        const ArrayDOFV<nVarsFixed> &u,
        real alphaDiag,
        real dt,
        int nNewtonSteps,
        SourceFilter filter,
        OptionalRef<ArrayDOFV<1>> cellTWarm)
    {
        DNDS_check_throw_info(dt > 0, "PointImplicitSourceUpdate requires positive dt");
        DNDS_check_throw_info(nNewtonSteps >= 0, "PointImplicitSourceUpdate requires non-negative nNewtonSteps");

        // Pseudo-time march C + alphaDiag*S(u) - u/dt to steady state.
        // The reactive split owns only transported rhoY equations; rho, momentum,
        // and rhoE remain fixed during this point solve.

        TDiffU zeroGrad;
        zeroGrad.resize(Eigen::NoChange, nVars);
        zeroGrad.setZero();

        uNew = u;
        const bool reactiveSpeciesOnly = filter == SourceFilter::ReactiveOnly && phys_.hasChemicalSource();
        const int Ns1Point = reactiveSpeciesOnly ? phys_.nSpecies() - 1 : nVars;
        const int iVarBegPoint = reactiveSpeciesOnly ? nVars - Ns1Point : 0;
        const int nPointVars = reactiveSpeciesOnly ? Ns1Point : nVars;
        const int nPseudoSteps = std::max(nNewtonSteps, 8);
        const real chemPseudoTimeScale = reactiveSpeciesOnly ? settings.reactiveFlow.CFLScale : real(1);
        const real chemPseudoTimeFloor = reactiveSpeciesOnly ? settings.reactiveFlow.chemRelaxEps : real(0);
        const real chemResidualAbsTol = reactiveSpeciesOnly ? settings.reactiveFlow.chemAbsTol : smallReal;
        DNDS_check_throw_info(std::isfinite(chemPseudoTimeScale) && chemPseudoTimeScale > 0,
                              "PointImplicitSourceUpdate requires positive finite reactiveFlow.CFLScale");
        DNDS_check_throw_info(std::isfinite(chemPseudoTimeFloor) && chemPseudoTimeFloor >= 0,
                              "PointImplicitSourceUpdate requires non-negative finite reactiveFlow.chemRelaxEps");
        DNDS_check_throw_info(std::isfinite(chemResidualAbsTol) && chemResidualAbsTol >= 0,
                              "PointImplicitSourceUpdate requires non-negative finite reactiveFlow.chemAbsTol");
        // Local experiment switch:
        // 0: local pseudo-time linearized march; 1: Cantera affine reactor;
        // 2: single linear source-Jacobian smoother, (I/dt + I/dtau + JS)^-1.
        static constexpr int pointImplicitSourceUpdatePath = 2;
        const bool outputResidualRatio = settings.pointImplicitSourceUpdateOut == 1;
        std::vector<real> localRatioMin(std::max(nPseudoSteps, 0) * nVars, veryLargeReal);
        std::vector<real> localRatioMax(std::max(nPseudoSteps, 0) * nVars, 0.0);
        auto repairReactiveSpecies = [&](TU &state)
        {
            if (!phys_.hasChemicalSource())
                return;
            int Ns = phys_.nSpecies();
            int Ns1 = Ns - 1;
            int nV = static_cast<int>(state.size());
            int Isp = nV - Ns1;
            constexpr real rhoYFloor = 0.0;
            real rhoEBaseBeforeClip = phys_.mixtureBaseInternalRhoERaw(state);
            bool speciesClipped = false;
            for (int k = 0; k < Ns1; ++k)
            {
                real rhoYOld = state(Isp + k);
                state(Isp + k) = std::max(rhoYOld, rhoYFloor);
                speciesClipped = speciesClipped || (state(Isp + k) != rhoYOld);
            }
            real sumRhoY = 0;
            for (int k = 0; k < Ns1; ++k)
                sumRhoY += state(Isp + k);
            if (sumRhoY > state(0))
            {
                real scale = state(0) / sumRhoY * (1.0 - 1e-14);
                for (int k = 0; k < Ns1; ++k)
                    state(Isp + k) *= scale;
                speciesClipped = true;
            }
            if (speciesClipped)
            {
                real rhoEBaseAfterClip = phys_.mixtureBaseInternalRhoERaw(state);
                state(I4) += rhoEBaseAfterClip - rhoEBaseBeforeClip;
            }
        };
        auto validPointSourceState = [&](const TU &state) -> bool
        {
            if (!state.allFinite() || state(0) <= 0)
                return false;
            real rhoInv = 1.0 / state(0);
            real kinetic = 0;
            for (int iDim = 1; iDim <= dim; iDim++)
                kinetic += state(iDim) * state(iDim);
            kinetic *= 0.5 * rhoInv;
            real rhoESensible = state(I4) - kinetic - phys_.mixtureBaseInternalRhoE(state);
            if (!std::isfinite(rhoESensible) || rhoESensible <= 0)
                return false;
            try
            {
                real T = phys_.temperature(state);
                real p = state(0) * phys_.Rgas(state) * T;
                return std::isfinite(T) && std::isfinite(p) && p > 0 && phys_.toPhysT(T) >= phys_.temperatureFloor();
            }
            catch (const std::exception &)
            {
                return false;
            }
        };
        auto activeResidualNorm = [&](const TU &residualIn) -> real
        {
            real ret = 0;
            for (int iVar = iVarBegPoint; iVar < iVarBegPoint + nPointVars; iVar++)
                ret = std::max(ret, std::abs(residualIn(iVar)));
            return ret;
        };
        auto recordResidualRatio = [&](int iPseudo, const TU &residualIn, const TU &res0Abs)
        {
            if (!outputResidualRatio)
                return;
            TU ratio = residualIn.cwiseAbs().cwiseQuotient(res0Abs);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp critical(DNDSPointImplicitSourceRatio)
#endif
            {
                for (int iVar = 0; iVar < nVars; iVar++)
                {
                    if (reactiveSpeciesOnly && iVar < iVarBegPoint)
                        ratio(iVar) = 0.0;
                    const index iOut = static_cast<index>(iPseudo) * nVars + iVar;
                    localRatioMin[iOut] = std::min(localRatioMin[iOut], ratio(iVar));
                    localRatioMax[iOut] = std::max(localRatioMax[iOut], ratio(iVar));
                }
            }
        };
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(guided)
#endif
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            TU sourceAtU;
            sourceAtU.setZero(nVars);
            TJacobianU sourceJac;
            EvaluateCellSource(sourceAtU, sourceJac, u[iCell], zeroGrad,
                               iCell, 0, filter, 1.0, false, {}, {}, {}, true, 0, cellTWarm);
            auto refreshCellTWarm = [&](const TU &state)
            {
                if (!cellTWarm || !phys_.hasChemicalSource())
                    return;
                (*cellTWarm)[iCell](0) = phys_.temperature(state, (*cellTWarm)[iCell](0));
            };
            auto sourceResidual = [&](TU &residualOut, const TU &state, const TU &sourceAtState)
            {
                residualOut = res[iCell];
                residualOut -= alphaDiag * sourceAtU;
                residualOut += alphaDiag * sourceAtState;
                residualOut += (1.0 / dt) * u[iCell];
                residualOut -= (1.0 / dt) * state;
            };
            TU res0Abs = res[iCell].cwiseAbs();
            for (int iVar = 0; iVar < nVars; iVar++)
                if (!reactiveSpeciesOnly || iVar >= iVarBegPoint)
                    res0Abs(iVar) += smallReal;
                else
                    res0Abs(iVar) = 1.0;

            if constexpr (pointImplicitSourceUpdatePath == 2)
            {
                // Recompute JS locally for now. This could be saved directly from
                // the manual frhs/EvaluateRHS source-Jacobian assembly site, but
                // doing so needs clearer control wiring for the split-source path.
                TU sourceAtUNew;
                sourceAtUNew.setZero(nVars);
                sourceJac.setZero(nVars, nVars);
                EvaluateCellSource(sourceAtUNew, sourceJac, uNew[iCell], zeroGrad,
                                   iCell, 2, filter, 1.0, false, {}, {}, {}, true, 0, cellTWarm);

                // This uses the real rebuilt residual to preserve the fixed point
                // of the full dual-time system. Consequently, this split form is
                // ill-formed in the S -> 0 limit: it does not recover identity even
                // when reactiveSourceScale is zero.
                TU residualLocal;
                sourceResidual(residualLocal, uNew[iCell], sourceAtUNew);
                real sourceJacScale = 0.0;
                for (int iRow = iVarBegPoint; iRow < iVarBegPoint + nPointVars; iRow++)
                    for (int iCol = iVarBegPoint; iCol < iVarBegPoint + nPointVars; iCol++)
                        sourceJacScale = std::max(sourceJacScale, std::abs(alphaDiag * sourceJac(iRow, iCol)));
                real pseudoDtau = std::max(chemPseudoTimeFloor,
                                           chemPseudoTimeScale / (1.0 / dt + sourceJacScale + smallReal));

                TU delta;
                delta.setZero(nVars);
                if (reactiveSpeciesOnly)
                {
                    Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> lhs =
                        alphaDiag * sourceJac(Eigen::seq(iVarBegPoint, EigenLast), Eigen::seq(iVarBegPoint, EigenLast));
                    lhs.diagonal().array() += 1.0 / dt + 1.0 / pseudoDtau;
                    Eigen::Vector<real, Eigen::Dynamic> rhs = residualLocal(Eigen::seq(iVarBegPoint, EigenLast));
                    delta(Eigen::seq(iVarBegPoint, EigenLast)) = lhs.partialPivLu().solve(rhs.eval());
                }
                else
                {
                    TJacobianU lhs = alphaDiag * sourceJac;
                    lhs.diagonal().array() += 1.0 / dt + 1.0 / pseudoDtau;
                    delta = lhs.partialPivLu().solve(residualLocal.eval());
                }

                TU candidate = uNew[iCell] + delta;
                repairReactiveSpecies(candidate);
                if (validPointSourceState(candidate))
                {
                    TU sourceAtCandidate;
                    sourceAtCandidate.setZero(nVars);
                    EvaluateCellSource(sourceAtCandidate, sourceJac, candidate, zeroGrad,
                                       iCell, 0, filter, 1.0, false, {}, {}, {}, true, 0, cellTWarm);
                    TU residualCandidate;
                    sourceResidual(residualCandidate, candidate, sourceAtCandidate);
                    uNew[iCell] = candidate;
                    refreshCellTWarm(uNew[iCell]);
                    for (int iPseudo = 0; iPseudo < nPseudoSteps; iPseudo++)
                        recordResidualRatio(iPseudo, residualCandidate, res0Abs);
                    continue;
                }
            }

            if constexpr (pointImplicitSourceUpdatePath == 1)
            {
                if (reactiveSpeciesOnly)
                {
                    try
                    {
                        std::vector<double> Y(static_cast<size_t>(phys_.nSpecies()), 0.0);
                        real rhoInv = 1.0 / std::max(uNew[iCell](0), real(1e-60));
                        real sumY = 0.0;
                        for (int k = 0; k < Ns1Point; ++k)
                        {
                            Y[static_cast<size_t>(k)] = std::max(real(0), uNew[iCell](iVarBegPoint + k) * rhoInv);
                            sumY += Y[static_cast<size_t>(k)];
                        }
                        Y[static_cast<size_t>(Ns1Point)] = std::max(real(0), 1.0 - sumY);

                        std::vector<double> constantTerm(static_cast<size_t>(phys_.nSpecies()), 0.0);
                        real sumC = 0.0;
                        for (int k = 0; k < Ns1Point; ++k)
                        {
                            int iVar = iVarBegPoint + k;
                            constantTerm[static_cast<size_t>(k)] =
                                (res[iCell](iVar) - alphaDiag * sourceAtU(iVar) + u[iCell](iVar) / dt) * rhoInv;
                            sumC += constantTerm[static_cast<size_t>(k)];
                        }
                        constantTerm[static_cast<size_t>(Ns1Point)] = 1.0 / dt - sumC;

                        TU candidate = uNew[iCell];
                        real T = phys_.temperature(candidate, cellTWarm ? (*cellTWarm)[iCell](0) : real(0));
                        phys_.advanceAffineConstVolumeY(
                            T, candidate(0),
                            Chemistry::SpeciesBufferView{Y.data(), static_cast<int>(Y.size())},
                            alphaDiag, dt,
                            Chemistry::ConstSpeciesBufferView{constantTerm.data(), static_cast<int>(constantTerm.size())},
                            1.0 * dt,
                            1e-4, 1e-4, 0, 2000);
                        for (int k = 0; k < Ns1Point; ++k)
                            candidate(iVarBegPoint + k) = candidate(0) * Y[static_cast<size_t>(k)];
                        repairReactiveSpecies(candidate);

                        if (validPointSourceState(candidate))
                        {
                            TU sourceAtCandidate;
                            sourceAtCandidate.setZero(nVars);
                            EvaluateCellSource(sourceAtCandidate, sourceJac, candidate, zeroGrad,
                                               iCell, 0, filter, 1.0, false, {}, {}, {}, true, 0, cellTWarm);
                            TU residualCandidate;
                            sourceResidual(residualCandidate, candidate, sourceAtCandidate);
                            uNew[iCell] = candidate;
                            refreshCellTWarm(uNew[iCell]);
                            for (int iPseudo = 0; iPseudo < nPseudoSteps; iPseudo++)
                                recordResidualRatio(iPseudo, residualCandidate, res0Abs);
                            continue;
                        }
                    }
                    catch (const std::exception &)
                    {
                        // Fall through to the local pseudo-time linearized update.
                    }
                }
            }

            real pseudoDtauScale = 1.0;
            for (int iPseudo = 0; iPseudo < nPseudoSteps; iPseudo++)
            {
                TU sourceAtUNew;
                sourceAtUNew.setZero(nVars);
                sourceJac.setZero(nVars, nVars);
                EvaluateCellSource(sourceAtUNew, sourceJac, uNew[iCell], zeroGrad,
                                   iCell, 2, filter, 1.0, false, {}, {}, {}, true, 0, cellTWarm);

                TU residualLocal;
                sourceResidual(residualLocal, uNew[iCell], sourceAtUNew);
                const real oldResidualNorm = activeResidualNorm(residualLocal);
                if (oldResidualNorm <= chemResidualAbsTol)
                {
                    recordResidualRatio(iPseudo, residualLocal, res0Abs);
                    break;
                }

                real sourceJacScale = 0.0;
                for (int iRow = iVarBegPoint; iRow < iVarBegPoint + nPointVars; iRow++)
                    for (int iCol = iVarBegPoint; iCol < iVarBegPoint + nPointVars; iCol++)
                        sourceJacScale = std::max(sourceJacScale, std::abs(alphaDiag * sourceJac(iRow, iCol)));
                real pseudoDtau = std::max(chemPseudoTimeFloor,
                                           chemPseudoTimeScale * pseudoDtauScale / (1.0 / dt + sourceJacScale + smallReal));

                // sourceJac stores -d(source)/dU. Therefore
                // (I / dtau + I / dt + alphaDiag * sourceJac) * delta = residualLocal.
                TU delta;
                delta.setZero(nVars);
                if (reactiveSpeciesOnly)
                {
                    Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> lhs =
                        alphaDiag * sourceJac(Eigen::seq(iVarBegPoint, EigenLast), Eigen::seq(iVarBegPoint, EigenLast));
                    lhs.diagonal().array() += 1.0 / dt + 1.0 / pseudoDtau;
                    Eigen::Vector<real, Eigen::Dynamic> rhs = residualLocal(Eigen::seq(iVarBegPoint, EigenLast));
                    delta(Eigen::seq(iVarBegPoint, EigenLast)) = lhs.partialPivLu().solve(rhs.eval());
                }
                else
                {
                    TJacobianU lhs = alphaDiag * sourceJac;
                    lhs.diagonal().array() += 1.0 / dt + 1.0 / pseudoDtau;
                    delta = lhs.partialPivLu().solve(residualLocal.eval());
                }
                TU accepted = uNew[iCell];
                TU acceptedResidual = residualLocal;
                real acceptedResidualNorm = oldResidualNorm;
                bool acceptedValid = false;
                for (int iRelax = 0; iRelax < 8; iRelax++)
                {
                    TU deltaTry = this->CompressInc(uNew[iCell], delta);
                    TU candidate = uNew[iCell] + deltaTry;
                    repairReactiveSpecies(candidate);
                    if (!validPointSourceState(candidate))
                    {
                        pseudoDtau *= 0.5;
                        delta *= 0.5;
                        continue;
                    }
                    TU sourceAtCandidate;
                    sourceAtCandidate.setZero(nVars);
                    EvaluateCellSource(sourceAtCandidate, sourceJac, candidate, zeroGrad,
                                       iCell, 0, filter, 1.0, false, {}, {}, {}, true, 0, cellTWarm);
                    TU residualCandidate;
                    sourceResidual(residualCandidate, candidate, sourceAtCandidate);
                    real candidateResidualNorm = activeResidualNorm(residualCandidate);
                    if (candidateResidualNorm <= oldResidualNorm)
                    {
                        accepted = candidate;
                        acceptedResidual = residualCandidate;
                        acceptedResidualNorm = candidateResidualNorm;
                        acceptedValid = true;
                        break;
                    }
                    pseudoDtau *= 0.5;
                    delta *= 0.5;
                }
                if (!acceptedValid)
                {
                    recordResidualRatio(iPseudo, residualLocal, res0Abs);
                    break;
                }
                uNew[iCell] = accepted;
                refreshCellTWarm(uNew[iCell]);
                recordResidualRatio(iPseudo, acceptedResidual, res0Abs);
                pseudoDtauScale = acceptedResidualNorm < oldResidualNorm * 0.8
                                      ? std::min(pseudoDtauScale * 1.5, 100.0)
                                      : std::max(pseudoDtauScale * 0.7, 1e-6);
            }
        }
        if (outputResidualRatio && nPseudoSteps > 0)
        {
            std::vector<real> ratioMin(localRatioMin.size(), 0.0);
            std::vector<real> ratioMax(localRatioMax.size(), 0.0);
            MPI::Allreduce(localRatioMin.data(), ratioMin.data(), localRatioMin.size(), DNDS_MPI_REAL, MPI_MIN, u.father->getMPI().comm);
            MPI::Allreduce(localRatioMax.data(), ratioMax.data(), localRatioMax.size(), DNDS_MPI_REAL, MPI_MAX, u.father->getMPI().comm);
            if (u.father->getMPI().rank == 0)
            {
                log() << std::scientific;
                // Print column label header (rho, rhoU, rhoV, (rhoW), rhoE, ..., species...)
                {
                    log() << "PointImplicitSourceUpdate pseudo col labels:";
                    for (int iVar = 0; iVar < nVars; iVar++)
                    {
                        if (iVar < I4)
                            log() << (iVar == 0 ? " rho" : iVar == 1 ? " rhoU"
                                                       : iVar == 2   ? " rhoV"
                                                                     : " rhoW");
                        else if (iVar == I4)
                            log() << " rhoE";
                        else if (phys_.hasChemicalSource() && iVar >= I4 + 1)
                        {
                            int Isp = nVars - (phys_.nSpecies() - 1);
                            if (iVar >= Isp && iVar - Isp < phys_.nSpecies() - 1)
                                log() << " " << phys_.speciesName(iVar - Isp);
                            else if (iVar < Isp)
                                log() << " v" << iVar;
                            else
                                log() << " " << phys_.speciesName(phys_.nSpecies() - 1);
                        }
                        else
                            log() << " v" << iVar;
                    }
                    log() << std::endl;
                }
                for (int iPseudo = 0; iPseudo < nPseudoSteps; iPseudo++)
                {
                    log() << "PointImplicitSourceUpdate pseudo [" << iPseudo + 1 << "] res/res0 min [";
                    for (int iVar = 0; iVar < nVars; iVar++)
                        log() << (iVar ? "," : "") << ratioMin[static_cast<index>(iPseudo) * nVars + iVar];
                    log() << "] max [";
                    for (int iVar = 0; iVar < nVars; iVar++)
                        log() << (iVar ? "," : "") << ratioMax[static_cast<index>(iPseudo) * nVars + iVar];
                    log() << "]" << std::endl;
                }
            }
        }
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>, )
    void EulerEvaluator<model>::ReactiveSourceConstVolumeStep(
        ArrayDOFV<nVarsFixed> &u,
        ArrayRECV<nVarsFixed> &uRec,
        real dt,
        real t,
        OptionalRef<ArrayDOFV<1>> cellTWarm)
    {
        (void)uRec;
        (void)t;
        DNDS_check_throw_info(dt >= 0, "ReactiveSourceConstVolumeStep requires non-negative dt");
        if (dt == 0)
            return;
        DNDS_check_throw_info(Traits::isExtended && settings.reactiveFlow.enabled && phys_.hasChemicalSource(),
                              "ReactiveSourceConstVolumeStep requires reactive extended Euler physics");
        DNDS_check_throw_info(std::isfinite(settings.reactiveSourceScale) && settings.reactiveSourceScale >= 0,
                              "ReactiveSourceConstVolumeStep requires non-negative finite reactiveSourceScale");
        if (settings.reactiveSourceScale == 0.0)
            return;

        const int Ns = phys_.nSpecies();
        const int Ns1 = Ns - 1;
        const int Isp = nVars - Ns1;
        DNDS_check_throw_info(Isp >= I4 + 1,
                              "ReactiveSourceConstVolumeStep requires Isp >= I4+1 (RANS gap tolerated, frozen)");

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(guided)
#endif
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            auto state = u[iCell];
            DNDS_check_throw_info(std::isfinite(state(0)) && state(0) > 0,
                                  fmt::format("ReactiveSourceConstVolumeStep invalid density at cell {}", iCell));
            const real rho = state(0);
            const real rhoInv = 1.0 / rho;

            std::vector<double> Y(static_cast<size_t>(Ns), 0.0);
            real sumY = 0.0;
            for (int k = 0; k < Ns1; ++k)
            {
                Y[static_cast<size_t>(k)] = std::max(real(0), state(Isp + k) * rhoInv);
                sumY += Y[static_cast<size_t>(k)];
            }
            Y[static_cast<size_t>(Ns1)] = std::max(real(0), 1.0 - sumY);
            real yNorm = 0.0;
            for (double y : Y)
                yNorm += y;
            DNDS_check_throw_info(yNorm > 0,
                                  fmt::format("ReactiveSourceConstVolumeStep zero species mass at cell {}", iCell));
            for (double &y : Y)
                y /= yNorm;

            real T = phys_.temperature(state, cellTWarm ? (*cellTWarm)[iCell](0) : real(0));
            if (cellTWarm)
                (*cellTWarm)[iCell](0) = T;
            phys_.advanceConstVolumeY(
                T, rho,
                Chemistry::SpeciesBufferView{Y.data(), Ns},
                settings.reactiveSourceScale,
                dt,
                settings.reactorStepSettings.rtol,
                settings.reactorStepSettings.atol,
                settings.reactorStepSettings.maxOrder,
                settings.reactorStepSettings.maxSteps);

            real sumYIndependent = 0.0;
            for (int k = 0; k < Ns1; ++k)
            {
                real yk = std::max(real(0), static_cast<real>(Y[static_cast<size_t>(k)]));
                sumYIndependent += yk;
                state(Isp + k) = rho * yk;
            }
            if (sumYIndependent >= 1.0)
            {
                const real scale = (1.0 - 1e-14) / sumYIndependent;
                for (int k = 0; k < Ns1; ++k)
                    state(Isp + k) *= scale;
            }

            real TCheck = phys_.temperature(state, T);
            if (cellTWarm)
                (*cellTWarm)[iCell](0) = TCheck;
            real pCheck = rho * phys_.Rgas(state) * TCheck;
            DNDS_check_throw_info(std::isfinite(TCheck) && std::isfinite(pCheck) &&
                                      phys_.toPhysT(TCheck) >= phys_.chem().baseTemperature() && pCheck > 0,
                                  fmt::format("ReactiveSourceConstVolumeStep invalid post-step state at cell {}: T={} K, p_code={}",
                                              iCell, phys_.toPhysT(TCheck), pCheck));
        }

        u.trans.startPersistentPull();
        u.trans.waitPersistentPull();
    }

    template <EulerModel model>
    /** @brief Accumulate time-averaged solution field using running weighted average.
     *
     *  Updates wAveraged as a running weighted average: wAveraged = (tCur*wAveraged + dt*w) / (tCur+dt).
     *
     *  @param w          Current primitive variable field.
     *  @param wAveraged  Running time-averaged field (input/output).
     *  @param dt         Time step weight for current sample.
     *  @param tCur       Accumulated averaging time (input/output, incremented by dt).
     */
    void EulerEvaluator<model>::TimeAverageAddition(ArrayDOFV<nVarsFixed> &w, ArrayDOFV<nVarsFixed> &wAveraged, real dt, real &tCur)
    {
        wAveraged *= (tCur / (tCur + dt + verySmallReal));
        wAveraged.addTo(w, (dt + verySmallReal) / (tCur + dt + verySmallReal));
        tCur += dt + verySmallReal;
    }

    template <EulerModel model>
    /** @brief Convert cell mean values from conservative to primitive variables.
     *
     *  @param u  Conservative variable DOF array (input).
     *  @param w  Primitive variable DOF array (output).
     */
    void EulerEvaluator<model>::MeanValueCons2Prim(ArrayDOFV<nVarsFixed> &u, ArrayDOFV<nVarsFixed> &w)
    {
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (index iCell = 0; iCell < u.Size(); iCell++)
        {
            real T_cell = phys_.temperature(u[iCell]);
            real gammaEq = phys_.gammaEq(T_cell, u[iCell]);
            TU out;
            Gas::IdealGasThermalConservative2Primitive<dim>(u[iCell], out, gammaEq,
                                                            phys_.mixtureBaseInternalRhoE(u[iCell]));
            w[iCell] = out;
        }
    }

    template <EulerModel model>
    /** @brief Convert cell mean values from primitive to conservative variables.
     *
     *  @param w  Primitive variable DOF array (input).
     *  @param u  Conservative variable DOF array (output).
     */
    void EulerEvaluator<model>::MeanValuePrim2Cons(ArrayDOFV<nVarsFixed> &w, ArrayDOFV<nVarsFixed> &u)
    {
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (index iCell = 0; iCell < w.Size(); iCell++)
        {
            TU out;
            phys_.primToConservative(w[iCell], out);
            u[iCell] = out;
        }
    }

    template <EulerModel model>
    /** @brief Evaluate the LP norm of the RHS residual (MPI-global).
     *
     *  For P < 3, computes component-wise LP norm; for P >= 3, computes L-infinity norm.
     *  Optionally volume-weighted and/or averaged.
     *
     *  @param res      Result vector of nVars component norms (output).
     *  @param rhs      Right-hand side residual.
     *  @param P        Norm exponent (1 for L1, 2 for L2, >=3 for L-infinity).
     *  @param volWise  If true, weight by cell volume.
     *  @param average  If true, divide by total weight.
     */
    void EulerEvaluator<model>::EvaluateNorm(Eigen::Vector<real, -1> &res, ArrayDOFV<nVarsFixed> &rhs, index P, bool volWise, bool average)
    {
        res.resize(nVars);
        if (P < 3)
        {
            TU resc;
            resc.setZero(nVars);
            real rescBase{0};
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp declare reduction(TUAdd:TU : omp_out += omp_in) initializer(omp_priv = omp_orig)
#    pragma omp parallel for schedule(static) reduction(TUAdd : resc)
#endif
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
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp declare reduction(TUAdd:TU : omp_out = omp_out.array().max(omp_in.array())) initializer(omp_priv = omp_orig)
#    pragma omp parallel for schedule(static) reduction(TUAdd : resc)
#endif
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                resc = resc.array().max(rhs[iCell].array().abs()).matrix();
            MPI::Allreduce(resc.data(), res.data(), res.size(), DNDS_MPI_REAL, MPI_MAX, rhs.father->getMPI().comm);
        }
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>, )
    void EulerEvaluator<model>::EvaluateMinMax(
        Eigen::Vector<real, -1> &uMin, Eigen::Vector<real, -1> &uMax, ArrayDOFV<nVarsFixed> &u,
        StateValueOrigin representation)
    {
        const int nV = nVars;
        uMin.resize(nV);
        uMax.resize(nV);
        TU localMin, localMax;
        localMin.setConstant(nV, std::numeric_limits<real>::max());
        localMax.setConstant(nV, std::numeric_limits<real>::lowest());

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp declare reduction(TUMin:TU : omp_out = omp_out.array().min(omp_in.array())) initializer(omp_priv = omp_orig)
#    pragma omp declare reduction(TUMax:TU : omp_out = omp_out.array().max(omp_in.array())) initializer(omp_priv = omp_orig)
#    pragma omp parallel for schedule(static) reduction(TUMin : localMin) reduction(TUMax : localMax)
#endif
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            if (representation == StateValueOrigin::Cons)
            {
                localMin = localMin.array().min(u[iCell].array()).matrix();
                localMax = localMax.array().max(u[iCell].array()).matrix();
            }
            else if (representation == StateValueOrigin::ConsPhy)
            {
                TU src = u[iCell];
                TU buf(nV);
                phys_.consCodeToPhys(src, buf);
                localMin = localMin.array().min(buf.array()).matrix();
                localMax = localMax.array().max(buf.array()).matrix();
            }
            else if (representation == StateValueOrigin::PrimTP)
            {
                TU buf(nV);
                phys_.conservativeToPrimTP(u[iCell], buf);
                localMin = localMin.array().min(buf.array()).matrix();
                localMax = localMax.array().max(buf.array()).matrix();
            }
            else if (representation == StateValueOrigin::PrimTPPhy)
            {
                TU buf(nV);
                TU tmp(nV);
                phys_.conservativeToPrimTP(u[iCell], tmp);
                phys_.primTPCodeToPhys(tmp, buf);
                localMin = localMin.array().min(buf.array()).matrix();
                localMax = localMax.array().max(buf.array()).matrix();
            }
            else
            {
                DNDS_assert_info(false, "EvaluateMinMax: unsupported StateValueOrigin");
                localMin = localMin.array().min(u[iCell].array()).matrix();
                localMax = localMax.array().max(u[iCell].array()).matrix();
            }
        }
        MPI::Allreduce(localMin.data(), uMin.data(), nV, DNDS_MPI_REAL, MPI_MIN, u.father->getMPI().comm);
        MPI::Allreduce(localMax.data(), uMax.data(), nV, DNDS_MPI_REAL, MPI_MAX, u.father->getMPI().comm);
    }

    template <class TU>
    struct TU_P_Reduction
    {
        TU _v;
        index P{1};

        void setZero(int n)
        {
            _v.setZero(n);
        }

        void reduce(const TU_P_Reduction<TU> &R)
        {
            _v = ((P < 3) ? TU(_v + R._v) : TU((_v.array().max(R._v.array())).matrix()));
        }
        void setP(index nP) { P = nP; }
    };

    template <EulerModel model>
    /** @brief Evaluate the LP norm of the reconstructed solution, optionally against a reference field.
     *
     *  Integrates the reconstructed solution (or its error against FCompareField) over
     *  quadrature points in each cell and reduces globally.
     *
     *  @param res               Result vector of nVars component norms (output).
     *  @param u                 Conservative variable DOF array.
     *  @param uRec              Reconstruction coefficients.
     *  @param P                 Norm exponent (1 for L1, 2 for L2, >=3 for L-infinity).
     *  @param compare           If true, compute error against FCompareField.
     *  @param FCompareField     Reference field function(point, time) -> TU.
     *  @param FCompareFieldWeight  Weight function(point, time) -> TU for error weighting.
     *  @param t                 Current simulation time.
     */
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
        TU_P_Reduction<TU> resc;
        resc.setZero(nVars);
        resc.setP(P);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp declare reduction(TUAdd : TU_P_Reduction<TU> : omp_out.reduce(omp_in)) initializer(omp_priv = omp_orig)
#    pragma omp parallel for schedule(static) reduction(TUAdd : resc)
#endif
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
                        resc._v = resc._v.array().max(uR.array().abs());
                    inc = uR.array().abs().pow(P);
                    inc *= vfv->GetCellJacobiDet(iCell, iG);
                });
            if (P < 3)
                resc._v += rescCell;
        }
        if (P > 3)
            MPI::Allreduce(resc._v.data(), res.data(), res.size(), DNDS_MPI_REAL, MPI_MAX, u.father->getMPI().comm);
        else
        {
            MPI::Allreduce(resc._v.data(), res.data(), res.size(), DNDS_MPI_REAL, MPI_SUM, u.father->getMPI().comm);
            res = res.array().pow(1.0 / P).matrix();
        }
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>, template <>)
    /** @brief Apply positivity-preserving gradient limiter to reconstructed solution gradients.
     *
     *  Limits the gradient so that density and internal energy remain above threshold
     *  values at all face quadrature points (Zhang-Shu style for gradients).
     *  Optionally disables shock-based limiting via flags.
     *
     *  @param u         Conservative variable DOF array.
     *  @param uGrad     Input gradient array.
     *  @param uGradNew  Limited gradient array (output).
     *  @param flags     Bitfield flags (e.g., LIMITER_UGRAD_Disable_Shock_Limiter).
     */
    void EulerEvaluator<model>::LimiterUGrad(
        ArrayDOFV<nVarsFixed> &u, ArrayGRADV<nVarsFixed, gDim> &uGrad, ArrayGRADV<nVarsFixed, gDim> &uGradNew,
        uint64_t flags)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        static constexpr real safetyRatio = 1 - 1e-2;
        static constexpr real E_lb_eps = 1e-2;
        static constexpr real ratio_decay = 0.75;

        bool disable_shock_limiter = flags & LIMITER_UGRAD_Disable_Shock_Limiter;

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            uGradNew[iCell] = uGrad[iCell];
            auto c2f = mesh->cell2face[iCell];

            TU_Batch uFaceInc;
            uFaceInc.setZero(nVars, c2f.size() * 2); // j < c2f.size(): faceInc; j > c2f.size(): baryInc
            TU uOtherMin = u[iCell];
            TU uOtherMax = u[iCell];
            auto fEInternal = [this](const TU &u) -> real
            { return u(I4) - 0.5 * u(Seq123).squaredNorm() / (u(0) + verySmallReal) - phys_.mixtureBaseInternalRhoE(u); };
            real eOtherMin{0}, eOtherMax{0};
            eOtherMin = eOtherMax = fEInternal(u[iCell]);
            for (rowsize ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto if2c = mesh->CellIsFaceBack(iCell, iFace, ic2f) ? 0 : 1;
                uFaceInc(EigenAll, ic2f) =
                    uGrad[iCell].transpose() *
                    (vfv->GetFaceQuadraturePPhysFromCell(iFace, iCell, if2c, -1) - vfv->GetCellQuadraturePPhys(iCell, -1))(SeqG012);
                index iCellOther = mesh->CellFaceOther(iCell, iFace, ic2f);
                if (iCellOther != UnInitIndex)
                {
                    uOtherMin = uOtherMin.array().min(u[iCellOther].array());
                    uOtherMax = uOtherMax.array().max(u[iCellOther].array());
                    eOtherMin = std::min(eOtherMin, fEInternal(u[iCellOther]));
                    eOtherMax = std::max(eOtherMax, fEInternal(u[iCellOther]));
                    // ! adding bary value if the other cell exists
                    // uFaceInc(EigenAll, ic2f + c2f.size()) =
                    // uGrad[iCell].transpose() *
                    //  (vfv->GetOtherCellBaryFromCell(iCell, iCellOther, -1) - vfv->GetCellQuadraturePPhys(iCell, -1))(SeqG012);
                }
            }

            TU uFaceIncMax = uFaceInc.array().rowwise().maxCoeff().array().max(0.0).matrix();
            TU uFaceIncMin = uFaceInc.array().rowwise().minCoeff().array().min(0.0).matrix();
            if (!disable_shock_limiter)
            {
                TU alpha0;
                alpha0.setConstant(nVars, 1.0);
                alpha0 = alpha0.array().min(((uOtherMax - u[iCell]).array() / (uFaceIncMax.array() + verySmallReal)));
                alpha0 = alpha0.array().min(((u[iCell] - uOtherMin).array() / (-uFaceIncMin.array() + verySmallReal)));
                uGradNew[iCell].array().rowwise() *= alpha0.array().transpose();
                uFaceInc.array().colwise() *= alpha0.array();
            }

            // start PP
            real alphaPP_Rho = 1.0;
            if (disable_shock_limiter) // do rho PP first
            {
                alphaPP_Rho = std::min(alphaPP_Rho, u[iCell][0] / (std::abs(uFaceIncMin(0)) + smallReal * u[iCell][0]));
                if (alphaPP_Rho < 1.0)
                    alphaPP_Rho *= safetyRatio;
            }

            TU_Batch uFaceAlpha0 = (uFaceInc * alphaPP_Rho).colwise() + u[iCell];
            for (int j = 0; j < uFaceAlpha0.cols(); j++)
                DNDS_assert(uFaceAlpha0(0, j) > 0);
            real minEFace = veryLargeReal;
            for (int j = 0; j < uFaceAlpha0.cols(); j++)
                minEFace = std::min(minEFace, fEInternal(uFaceAlpha0(EigenAll, j)));
            real eC = fEInternal(u[iCell]);
            real deltaEFaceMin = minEFace - eC;
            real alphaPP_E = 1.0;
            if (deltaEFaceMin < 0)
                alphaPP_E = std::min(alphaPP_E, std::abs(eC * (1 - E_lb_eps)) / (verySmallReal - deltaEFaceMin));
            if (alphaPP_E < 1.0)
                alphaPP_E *= safetyRatio;

            int i_decay = 0;
            for (; i_decay < 1000; i_decay++)
            {
                uFaceAlpha0 = (uFaceInc * (alphaPP_Rho * alphaPP_E)).colwise() + u[iCell];
                real minEFaceDecay = veryLargeReal;
                for (int j = 0; j < uFaceAlpha0.cols(); j++)
                    minEFaceDecay = std::min(minEFaceDecay, fEInternal(uFaceAlpha0(EigenAll, j)));
                if (minEFaceDecay >= eC * (1 - E_lb_eps))
                    break;
                alphaPP_E *= ratio_decay;
            }
            if (i_decay >= 1000)
            {
                std::ostringstream oss;
                oss << std::scientific << std::setprecision(4);
                oss << "LimiterUGrad: species PP decay exhausted on cell " << iCell << "\n";
                oss << "  alphaPP_Rho=" << alphaPP_Rho << " alphaPP_E=" << alphaPP_E
                    << " eC=" << eC << " E_lb_eps=" << E_lb_eps << "\n";
                oss << "  u[cell] = " << u[iCell].transpose() << "\n";
                oss << "  uGrad[cell] = \n"
                    << uGrad[iCell] << "\n";
                oss << "  uFaceInc (nVars x nFace) = \n"
                    << uFaceInc << "\n";
                oss << "  The sensible-energy positivity check against mixtureBaseInternalRhoE\n"
                    << "  could not be satisfied even after 1000 decay iterations.  This\n"
                    << "  indicates the cell-mean itself has invalid species that the\n"
                    << "  gradient limiter cannot repair.";
                DNDS_assert_info(false, oss.str());
            }

            uGradNew[iCell](EigenAll, Seq01234) *= alphaPP_Rho * alphaPP_E;
            if constexpr (Traits::isExtended)
            {
                if (phys_.hasChemicalSource())
                {
                    int Ns1 = phys_.nSpecies() - 1;
                    int Isp = nVars - Ns1;
                    uGradNew[iCell](EigenAll, Eigen::seq(Isp, EigenLast)) *= alphaPP_Rho * alphaPP_E;
                }
            }
        }
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>, template <>)
    /** @brief Evaluate the positivity-preserving reconstruction limiter coefficient (beta) per cell.
     *
     *  Computes a scalar beta in [0,1] for each cell such that the reconstructed
     *  solution at all face and (optionally) volume quadrature points maintains
     *  positive density and sensible internal energy. Uses bisection to find the maximum allowable beta.
     *  Reports the total number of limited cells and the global minimum beta.
     *
     *  @param u         Conservative variable DOF array.
     *  @param uRec      Reconstruction coefficients (may be modified for order reduction).
     *  @param uRecBeta  Per-cell limiter coefficient (output).
     *  @param nLim      Total number of limited cells across all MPI ranks (output).
     *  @param betaMin   Global minimum beta value (output).
     *  @param flag      Evaluation mode flags.
     */
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
            real rhoEiMin = veryLargeReal;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime) reduction(min : rhoMin, rhoEiMin)
#endif
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                rhoMin = std::min(rhoMin, u[iCell](0));
                real rhoEi_cell = u[iCell](I4) - 0.5 * u[iCell](Seq123).squaredNorm() / u[iCell](0) - phys_.mixtureBaseInternalRhoE(u[iCell]);
                rhoEiMin = std::min(rhoEiMin, rhoEi_cell);
            }
            MPI::AllreduceOneReal(rhoMin, MPI_MIN, mesh->getMPI());
            MPI::AllreduceOneReal(rhoEiMin, MPI_MIN, mesh->getMPI());
            rhoEps = std::min(rhoEps, minRatio * rhoMin);
            pEps = std::min(pEps, minRatio * rhoEiMin);
        }
        real rhoeSensibleEps = pEps;

        index nLimLocal = 0;
        real minBetaLocal = 1;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime) reduction(+ : nLimLocal) reduction(min : minBetaLocal)
#endif
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
                    quadBase(iG, EigenAll) = vfv->GetIntPointDiffBaseValue(iCell, -1, -1, iG, 0, 1);
                nPoint += gCell.GetNumPoints();
            }
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                auto iFace = c2f[ic2f];
                auto if2c = mesh->CellIsFaceBack(iCell, iFace, ic2f) ? 0 : 1;
                auto gFace = vfv->GetFaceQuad(iFace);
                for (int iG = 0; iG < gFace.GetNumPoints(); iG++)
                    quadBase(nPoint + iG, EigenAll) = vfv->GetIntPointDiffBaseValue(iCell, iFace, if2c, iG, 0, 1);
                nPoint += gFace.GetNumPoints();
            }
            /***********/
            DNDS_assert_info(u[iCell](0) >= rhoEps, fmt::format("rhoMean {}, {}", u[iCell](0), rhoEps));
            real T_cell = phys_.temperature(u[iCell]);
            // real gamma = phys_.gammaEq(T_cell, u[iCell]);
            real rhoE_base_cell = phys_.mixtureBaseInternalRhoE(u[iCell]);
            real rhoeSensibleCent = u[iCell](I4) - 0.5 * u[iCell](Seq123).squaredNorm() / u[iCell](0) - rhoE_base_cell;
            DNDS_assert_info(rhoeSensibleCent >= rhoeSensibleEps, fmt::format("rhoeSensibleMean {}, {}", rhoeSensibleCent, rhoeSensibleEps));

            if constexpr (Traits::isExtended)
            {
                if (phys_.hasChemicalSource())
                {
                    int Ns1 = phys_.nSpecies() - 1;
                    int Isp = nVars - Ns1;
                    for (int k = 0; k < Ns1; ++k)
                    {
                        real rhoYk = u[iCell](Isp + k);
                        DNDS_assert_info(rhoYk >= 0,
                                         fmt::format("cell mean rhoY[{}] = {:.3e} < 0", k, rhoYk));
                    }
                    DNDS_assert_info(u[iCell](Eigen::seqN(Isp, Ns1)).sum() <= u[iCell](0) + smallReal * u[iCell](0),
                                     "sumRhoY exceeds rho at cell mean");
                }
            }

            auto rhoE_base_perQ = [&](const Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> &rec)
            {
                Eigen::Vector<real, Eigen::Dynamic> hf(rec.rows());
                for (int ig = 0; ig < rec.rows(); ++ig)
                {
                    typename EulerEvaluator<model>::TU tmp = rec.row(ig);
                    hf(ig) = phys_.mixtureBaseInternalRhoERaw(tmp);
                }
                return hf;
            };

            // alter uRec if necessary
            int curOrder = vfv->GetCellOrder(iCell);
            Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> uRecBase = uRec[iCell];
            Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> recBase; // * has to call checkRecBaseGood() to hold valid value
            auto checkRecBaseGood = [&]() -> bool
            {
                recBase = (quadBase * uRecBase).rowwise() + u[iCell].transpose();
                if (recBase(EigenAll, 0).minCoeff() < rhoEps) // TODO: add relaxation to eps values
                    return false;
                if constexpr (Traits::hasSA)
                    if (recBase(EigenAll, I4 + 1).minCoeff() < rhoEps)
                        return false;
                if constexpr (Traits::has2EQ)
                    if (recBase(EigenAll, I4 + 1).minCoeff() < rhoEps || recBase(EigenAll, I4 + 2).minCoeff() < rhoEps)
                        return false;
                Eigen::Vector<real, Eigen::Dynamic> ek =
                    0.5 * (recBase(EigenAll, Seq123).array().square().rowwise().sum()) / recBase(EigenAll, 0).array();
                Eigen::Vector<real, Eigen::Dynamic> rhoE_base_q = rhoE_base_perQ(recBase);
                Eigen::Vector<real, Eigen::Dynamic> eInternalS = (recBase(EigenAll, I4) - ek - rhoE_base_q);
                if (eInternalS.minCoeff() < rhoeSensibleEps)
                    return false;
                // Species positivity: required because mixtureBaseInternalRhoE (used by
                // gammaEq in fluxFace) clips negative rhoY_k, increasing base energy and
                // lowering sensible energy relative to mixtureBaseInternalRhoERaw.
                if constexpr (Traits::isExtended)
                {
                    if (phys_.hasChemicalSource())
                    {
                        int Ns1 = phys_.nSpecies() - 1;
                        int Isp = nVars - Ns1;
                        auto speciesBlock = recBase(EigenAll, Eigen::seqN(Isp, Ns1));
                        if (speciesBlock.minCoeff() < 0)
                            return false;
                        if ((speciesBlock.rowwise().sum().array() > recBase(EigenAll, 0).array()).any())
                            return false;
                    }
                }
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
            Eigen::Vector<real, Eigen::Dynamic> rhoS = recInc(EigenAll, 0) + recBase(EigenAll, 0);
            Eigen::Index rhoMinIdx{-1};
            real rhoMin = rhoS.minCoeff(&rhoMinIdx);
            real theta1 = 1;
            if (rhoMin < rhoEps)
                for (int iG = 0; iG < rhoS.size(); iG++)
                    if (recInc(iG, 0) < 0) // negative increment
                        theta1 = std::min(theta1, (recBase(iG, 0) - rhoEps) / (-recInc(iG, 0) + verySmallReal));
#ifdef USE_NS_SA_NUT_REDUCED_ORDER
            if constexpr (Traits::hasSA)
            {
                static real v1Eps = smallReal * settings.refUPrim(I4 + 1);
                Eigen::Vector<real, Eigen::Dynamic> v1S = recInc(EigenAll, I4 + 1) + recBase(EigenAll, I4 + 1);
                real v1Min = v1S.minCoeff();
                if (v1Min < v1Eps)
                    for (int iG = 0; iG < rhoS.size(); iG++)
                        if (recInc(iG, I4 + 1) < 0) // negative increment
                            theta1 = std::min(theta1, (recBase(iG, I4 + 1) - v1Eps) / (-recInc(iG, I4 + 1) + verySmallReal))
                                // * 0 // to gain fully reduced order
                                ;
            }
#endif
            if constexpr (Traits::has2EQ)
            {
                real thetaC = 1;
                static real v1Eps = smallReal * settings.refUPrim(I4 + 1);
                static real v2Eps = smallReal * settings.refUPrim(I4 + 2);
                Eigen::Vector<real, Eigen::Dynamic> v1S = recInc(EigenAll, I4 + 1) + recBase(EigenAll, I4 + 1);
                Eigen::Vector<real, Eigen::Dynamic> v2S = recInc(EigenAll, I4 + 2) + recBase(EigenAll, I4 + 2);
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
                    uRec[iCell](EigenAll, {I4 + 1, I4 + 2}) *= thetaC * 0.9;
                }
            }

            // only compresses main flow part and rhoY_i species, avoiding RANS part
            recInc(EigenAll, Seq01234) *= theta1;
            if constexpr (Traits::isExtended)
            {
                if (phys_.hasChemicalSource())
                {
                    int Ns1 = phys_.nSpecies() - 1;
                    int Isp = nVars - Ns1;
                    recInc(EigenAll, Eigen::seq(Isp, EigenLast)) *= theta1;
                }
            }

            Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed>
                recVRhoG = recInc + recBase;

            Eigen::Vector<real, Eigen::Dynamic> ek = 0.5 * (recVRhoG(EigenAll, Seq123).array().square().rowwise().sum()) / recVRhoG(EigenAll, 0).array();
            Eigen::Vector<real, Eigen::Dynamic> rhoE_base_VR = rhoE_base_perQ(recVRhoG);
            Eigen::Vector<real, Eigen::Dynamic> eInternalS = recVRhoG(EigenAll, I4) - ek - rhoE_base_VR;
            real thetaP = 1.0;
            Eigen::Vector<real, Eigen::Dynamic> rhoE_base_B = rhoE_base_perQ(recBase);

            if (rhoeSensibleCent <= 2 * rhoeSensibleEps)
                thetaP = 0;
            else
                for (int iG = 0; iG < rhoS.size(); iG++)
                {
                    if (eInternalS(iG) < 2 * rhoeSensibleEps)
                    {
                        real thetaThis = Gas::IdealGasGetCompressionRatioPressure<dim, 1, nVarsFixed>(
                            recBase(iG, EigenAll).transpose(), recInc(iG, EigenAll).transpose(),
                            rhoeSensibleEps, rhoE_base_B(iG), rhoE_base_VR(iG));
                        thetaP = std::min(thetaP, thetaThis);
                    }
                }

            // --- species-aware PP check ---
            // thetaP was computed using raw base energy (mixtureBaseInternalRhoERaw).
            // gammaEq in fluxFace uses mixtureBaseInternalRhoE which clips negative
            // rhoY_k to 0, potentially increasing rhoE_base and making e_sensible go
            // negative.  Post-check using mixtureBaseInternalRhoE directly so the
            // computed sensible energy matches what gammaEq will see.  If violated,
            // zero the entire reconstruction increment (thetaP = 0), forcing
            // fallback to uRecBase which passed checkRecBaseGood.
            if constexpr (Traits::isExtended)
            {
                if (phys_.hasChemicalSource())
                {
                    Eigen::Vector<real, Eigen::Dynamic> ekC =
                        0.5 * (recVRhoG(EigenAll, Seq123).array().square().rowwise().sum()) / recVRhoG(EigenAll, 0).array();
                    Eigen::Vector<real, Eigen::Dynamic> rhoE_base_VC(nPoint);
                    for (int iG = 0; iG < nPoint; ++iG)
                    {
                        typename EulerEvaluator<model>::TU tmp = recVRhoG.row(iG);
                        rhoE_base_VC(iG) = phys_.mixtureBaseInternalRhoE(tmp);
                    }
                    Eigen::Vector<real, Eigen::Dynamic> eInternalSC =
                        recVRhoG(EigenAll, I4) - ekC - rhoE_base_VC;

                    if (eInternalSC.minCoeff() < rhoeSensibleEps)
                        thetaP = 0;
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
            // only compresses main flow part and rhoY_i species, avoiding RANS part
            if (uRecBeta[iCell](0) < 1)
            {
                uRec[iCell](EigenAll, Seq01234) = (uRec[iCell](EigenAll, Seq01234) - uRecBase(EigenAll, Seq01234)) * uRecBeta[iCell](0) + uRecBase(EigenAll, Seq01234);
                if constexpr (Traits::isExtended)
                {
                    if (phys_.hasChemicalSource())
                    {
                        int Ns1 = phys_.nSpecies() - 1;
                        int Isp = nVars - Ns1;
                        auto seqSpecies = Eigen::seq(Isp, EigenLast);
                        uRec[iCell](EigenAll, seqSpecies) = (uRec[iCell](EigenAll, seqSpecies) - uRecBase(EigenAll, seqSpecies)) * uRecBeta[iCell](0) + uRecBase(EigenAll, seqSpecies);
                    }
                }
            }

            // validation:
            recInc = quadBase * uRec[iCell];
            recVRhoG = recInc.rowwise() + u[iCell].transpose();
            ek = 0.5 * (recVRhoG(EigenAll, Seq123).array().square().rowwise().sum()) / recVRhoG(EigenAll, 0).array();
            rhoE_base_VR = rhoE_base_perQ(recVRhoG);
            eInternalS = (recVRhoG(EigenAll, I4) - ek - rhoE_base_VR);
            for (int iG = 0; iG < eInternalS.size(); iG++)
            {
                if (eInternalS(iG) < rhoeSensibleEps)
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
        real rhoeSensibleEps = pEps;
        bool ret{true};
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            real T_cell = phys_.temperature(u[iCell]);
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
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp critical
#endif
                ret = false;
            }
            real rhoeSensible = u[iCell](I4) - 0.5 * u[iCell](Seq123).squaredNorm() / u[iCell](0) - phys_.mixtureBaseInternalRhoE(u[iCell]);
            if (rhoeSensible < rhoeSensibleEps)
            {
                if (panic)
                    DNDS_assert_info(
                        false,
                        fmt::format(
                            "AssertMeanValuePP Failed on cell {} rhoeSensible\n",
                            iCell) +
                            fmt::format(
                                " eps={}, value={}",
                                rhoeSensibleEps, rhoeSensible));
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp critical
#endif
                ret = false;
            }

            // TODO: reactivate this
            // --- Species positivity assertion (reactive flow) ---
            if (phys_.hasChemicalSource())
            {
                int Ns = phys_.nSpecies();
                int Ns1 = Ns - 1;
                int nV = nVars;
                int Isp = nV - Ns1;

                for (int k = 0; k < Ns1; ++k)
                {
                    if (u[iCell](Isp + k) < 0)
                    {
                        if (panic)
                            DNDS_assert_info(
                                false,
                                fmt::format(
                                    "AssertMeanValuePP Failed on cell {} rhoY_{}\n",
                                    iCell, k) +
                                    fmt::format(
                                        " value={}",
                                        u[iCell](Isp + k)));
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp critical
#endif
                        ret = false;
                    }
                }

                real sumRhoY = 0;
                for (int k = 0; k < Ns1; ++k)
                    sumRhoY += u[iCell](Isp + k);
                if (sumRhoY > u[iCell](0))
                {
                    if (panic)
                        DNDS_assert_info(
                            false,
                            fmt::format(
                                "AssertMeanValuePP Failed on cell {} rhoY_last (dependent)\n",
                                iCell) +
                                fmt::format(
                                    " rho={}, sumRhoY={}",
                                    u[iCell](0), sumRhoY));
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp critical
#endif
                    ret = false;
                }
            }
        }

        return ret;
    }

    template <EulerModel model>
    /** @brief Evaluate per-cell RHS scaling factor (alpha) for positivity-preserving time stepping.
     *
     *  Determines the largest alpha in [0,1] such that u + alpha*res remains physically
     *  realizable (positive density and pressure) at all quadrature points. Used for
     *  under-relaxation to preserve positivity during explicit or implicit updates.
     *
     *  NOTE: uses mixtureBaseInternalRhoE (not Raw) for the sensible-energy
     *  check, matching gammaEq's species clipping in fluxFace.
     *
     *
     *  @param u             Conservative variable DOF array.
     *  @param uRec          Reconstruction coefficients.
     *  @param uRecBeta      Per-cell reconstruction limiter coefficient.
     *  @param res            RHS residual to be scaled.
     *  @param cellRHSAlpha  Per-cell alpha scaling factor (output).
     *  @param nLim          Total number of limited cells (output, MPI-reduced).
     *  @param alphaMin      Global minimum alpha (output, MPI-reduced).
     *  @param relax         Relaxation factor applied to alpha.
     *  @param compress      Compression mode for reconstruction part.
     *  @param flag          Evaluation mode flags (e.g., EvaluateCellRHSAlpha_MIN_ALL).
     */
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
        real rhoeSensibleEps = pEps;

        index nLimLocal = 0;
        real alphaMinLocal = 1;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime) reduction(+ : nLimLocal) reduction(min : alphaMinLocal)
#endif
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            real T_cell = phys_.temperature(u[iCell]);
            // real gamma = phys_.gammaEq(T_cell, u[iCell]);
            real alphaRho = 1;
            TU inc = res[iCell];
            DNDS_assert(u[iCell](0) >= rhoEps);
            real relaxedRho = rhoEps * relax + (u[iCell](0)) * (1 - relax);
            if (inc(0) < 0) // not < rhoEps!!!
                alphaRho = std::min(1.0, (u[iCell](0) - relaxedRho) / (-inc(0) - smallReal * inc(0)));
            DNDS_assert(alphaRho >= 0 && alphaRho <= 1);
            if constexpr (Traits::hasSA)
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
            if constexpr (Traits::has2EQ)
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
            real rhoE_base_new = phys_.mixtureBaseInternalRhoE(uNew);
            real rhoeSensibleNew = uNew(I4) - 0.5 * uNew(Seq123).squaredNorm() / uNew(0) - rhoE_base_new;
            real rhoeSensibleOld = u[iCell](I4) - 0.5 * u[iCell](Seq123).squaredNorm() / u[iCell](0) - phys_.mixtureBaseInternalRhoE(u[iCell]);
            real relaxedRhoeSensible = rhoeSensibleEps;
            if (rhoeSensibleNew < rhoeSensibleOld)
                relaxedRhoeSensible = rhoeSensibleEps + (rhoeSensibleOld - rhoeSensibleEps) * (1 - relax);

            real alphaP = 1;
            if (rhoeSensibleNew < relaxedRhoeSensible)
            {
                // todo: use high order accurate (add control switch)
                real alphaC = Gas::IdealGasGetCompressionRatioPressure<dim, 1, nVarsFixed>(
                    u[iCell], inc, relaxedRhoeSensible, phys_.mixtureBaseInternalRhoE(u[iCell]), rhoE_base_new);
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
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                if (cellRHSAlpha[iCell](0) < 1)
                    cellRHSAlpha[iCell](0) = alphaMin;
            }
        if (flag & EvaluateCellRHSAlpha_MIN_ALL)
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                cellRHSAlpha[iCell](0) = alphaMin;
    }

    template <EulerModel model>
    /** @brief Expand the per-cell alpha limiter to neighboring cells for smoothness.
     *
     *  Cells adjacent to limited cells (alpha < 1) are also given reduced alpha
     *  values to create a smooth transition region, preventing isolated sharp
     *  limiting boundaries.
     *
     *  @param u             Conservative variable DOF array.
     *  @param uRec          Reconstruction coefficients.
     *  @param uRecBeta      Per-cell reconstruction limiter coefficient.
     *  @param res            RHS residual.
     *  @param cellRHSAlpha  Per-cell alpha (input/output, expanded to neighbors).
     *  @param nLim          Additional limited cells from expansion (output, accumulated).
     *  @param alphaMin      Global minimum alpha (input).
     */
    void EulerEvaluator<model>::EvaluateCellRHSAlphaExpansion(
        ArrayDOFV<nVarsFixed> &u,
        ArrayRECV<nVarsFixed> &uRec,
        ArrayDOFV<1> &uRecBeta,
        ArrayDOFV<nVarsFixed> &res,
        ArrayDOFV<1> &cellRHSAlpha, index &nLim, real alphaMin)
    {
        static const real minRatio = 0.5;
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        real rhoEps = smallReal * settings.refUPrim(0) * 1e-1;
        real pEps = smallReal * settings.refUPrim(I4) * 1e-1;

        if (settings.ppEpsIsRelaxed)
        {
            real rhoMin = veryLargeReal;
            real rhoEiMin = veryLargeReal;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime) reduction(min : rhoMin, rhoEiMin)
#endif
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                rhoMin = std::min(rhoMin, u[iCell](0));
                real rhoEi_cell = u[iCell](I4) - 0.5 * u[iCell](Seq123).squaredNorm() / u[iCell](0) - phys_.mixtureBaseInternalRhoE(u[iCell]);
                rhoEiMin = std::min(rhoEiMin, rhoEi_cell);
            }
            MPI::AllreduceOneReal(rhoMin, MPI_MIN, mesh->getMPI());
            MPI::AllreduceOneReal(rhoEiMin, MPI_MIN, mesh->getMPI());
            rhoEps = std::min(rhoEps, minRatio * rhoMin);
            pEps = std::min(pEps, minRatio * rhoEiMin);
        }
        real rhoeSensibleEps = pEps;

        // Unused — kept for reference if alpha-expansion smoothing is ever re-enabled.
        // The calling loop at line ~2283 was commented out; see SEVERE #8-10 fixes
        // which replaced this expansion strategy with the current direct-limiting approach.
        auto cellIsHalfAlpha = [&](index iCell) -> bool // iCell should be internal
        {
            bool ret = false;
            if (cellRHSAlpha[iCell](0) == 1.0)
            {
                auto c2f = mesh->cell2face[iCell];
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iCellOther = vfv->CellFaceOther(iCell, c2f[ic2f], ic2f);
                    if (iCellOther != UnInitIndex)
                        if (cellRHSAlpha[iCellOther](0) != 1.0)
                            ret = true;
                }
            }
            return ret;
        };

        auto cellAdjAlphaMin = [&](index iCell) -> real // iCell should be internal (unused, same reason as cellIsHalfAlpha above)
        {
            real ret = 1;
            auto c2f = mesh->cell2face[iCell];
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iCellOther = vfv->CellFaceOther(iCell, c2f[ic2f], ic2f);
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
        // only check cells not already limited (α > alphaMin)
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            if (cellRHSAlpha[iCell](0) <= alphaMin)
                continue;

            TU inc = res[iCell];

            TU uNew = u[iCell] + inc;
            real rhoeSensibleNew = uNew(I4) - 0.5 * uNew(Seq123).squaredNorm() / uNew(0) - phys_.mixtureBaseInternalRhoE(uNew);

            if (rhoeSensibleNew < rhoeSensibleEps || uNew(0) < rhoEps)
            {
                cellRHSAlpha[iCell](0) = alphaMin;
                nLimLocal++;
            }
            if constexpr (Traits::hasSA)
            {
                // ** ! currently do not mass - fix for SA
                // static real v1Eps = smallReal * settings.refUPrim(I4 + 1);
                // if (uNew(I4 + 1) < v1Eps)
                //     cellRHSAlpha[iCell](0) = alphaMin;
            }
            if constexpr (Traits::has2EQ)
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
    /** @brief Smooth the local pseudo-time step by taking the minimum of the cell value and its neighbor average.
     *
     *  Prevents large time-step jumps between adjacent cells by capping each cell's
     *  dTau at the weighted average of its neighbors' values.
     *
     *  @param dTau     Input local pseudo-time step per cell.
     *  @param dTauNew  Smoothed pseudo-time step (output).
     */
    void EulerEvaluator<model>::MinSmoothDTau(
        ArrayDOFV<1> &dTau, ArrayDOFV<1> &dTauNew)
    {
        real smootherCentWeight = 1;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            auto c2f = mesh->cell2face[iCell];
            real nAdj = 0.;
            real dTMean = 0.;
            for (rowsize ic2f = 0; ic2f < c2f.size(); ++ic2f)
            {
                index iFace = c2f[ic2f];
                index iCellOther = vfv->CellFaceOther(iCell, iFace, ic2f);
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
    /** @brief Update boundary condition profiles from current solution for profile-anchored BCs.
     *
     *  For BC zones with anchorOpt==2, integrates the solution at boundary faces into
     *  a 1D radial profile recorder. Used for radial equilibrium pressure BCs in
     *  turbomachinery applications.
     *
     *  @param u     Conservative variable DOF array.
     *  @param uRec  Reconstruction coefficients.
     */
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
                    index iFace = mesh->bnd2faceV.at(iBnd);
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
                            real r = settings.frameConstRotation.rVec(coo(EigenAll, ic)).norm();
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
            index iFace = mesh->bnd2faceV.at(iBnd);
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
                real r = settings.frameConstRotation.rVec(coo(EigenAll, ic)).norm();
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
    /** @brief Compute radial equilibrium pressure distribution from BC profiles.
     *
     *  Integrates the centrifugal pressure gradient (rho * vt^2 / r) along the
     *  radial direction in boundary profiles to obtain the radial equilibrium
     *  pressure increment for turbomachinery outlet BCs.
     */
    void EulerEvaluator<model>::updateBCProfilesPressureRadialEq()
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
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
                        log() << fmt::format("EulerEvaluator<model>::updateBCProfilesPressureRadialEq: p rise: [{:.3e}]", v.second.v(I4, EigenLast))
                              << std::endl;
                }
            }
        }
    }
}
