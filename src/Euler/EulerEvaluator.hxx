/** @file EulerEvaluator.hxx
 *  @brief Template implementations of EulerEvaluator methods for implicit time-stepping,
 *         LU-SGS preconditioning, DOF initialization, boundary value generation,
 *         reconstruction limiting, norm evaluation, and BC profile updates.
 */
#pragma once

#include "DNDS/Core/Defines.hpp" // for correct  DNDS_SWITCH_INTELLISENSE
#include "Euler/EulerEvaluator.hpp"
#include "Geom/Compat/LegacyMeshRead.hpp"
#include "DNDS/Math/HardEigen.hpp"
#include <sstream>
#include <iomanip>
#include "Euler/SpecialFields.hpp"
#include "DNDS/Config/ExprtkWrapper.hpp"

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
        auto meshRead = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto meshLegacy = meshRead.LegacyHostView();
        const auto *mesh = &meshLegacy;
        this->ValidateMeshDependencies();
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
        auto meshRead = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto meshLegacy = meshRead.LegacyHostView();
        const auto *mesh = &meshLegacy;
        this->ValidateMeshDependencies();
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "LUSGSMatrixVec 1");
        int cnvars = nVars;

        auto cellOp = [&](index iCell) __attribute__((always_inline)){

        };

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (int iPart = 0; iPart < this->mesh->NLocalParts(); iPart++)
            for (const auto range : this->mesh->LocalPartRanges(iPart).Intervals())
                for (index iScan = range.begin; iScan < range.end; ++iScan)
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
        auto meshRead = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto meshLegacy = meshRead.LegacyHostView();
        const auto *mesh = &meshLegacy;
        this->ValidateMeshDependencies();
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "LUSGSMatrixToJacobianLU 1");
        int cnvars = nVars;
        jacLU.setZero();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (int iPart = 0; iPart < this->mesh->NLocalParts(); iPart++)
        {
            const auto worker = this->mesh->LocalPartRanges(iPart);
            for (const auto range : worker.Intervals())
                for (index iScan = range.begin; iScan < range.end; ++iScan)
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
                            worker.Contains(iCellOther))
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
                                for (int ic2c = 0; ic2c < this->mesh->LocalCellNeighbors()[iCell].size(); ic2c++)
                                    if (iCellOther == this->mesh->LocalCellNeighbors()[iCell][ic2c])
                                        iC2CInLocal = ic2c; // TODO: pre-search this
                                DNDS_assert(iC2CInLocal != -1);
                                jacLU.LDU(iCell, symLU->cell2cellFaceVLocal2FullRowPos[iCell][iC2CInLocal]) += jacBlock;
                            }
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
        auto meshRead = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto meshLegacy = meshRead.LegacyHostView();
        const auto *mesh = &meshLegacy;
        this->ValidateMeshDependencies();
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
        auto meshRead = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto meshLegacy = meshRead.LegacyHostView();
        const auto *mesh = &meshLegacy;
        this->ValidateMeshDependencies();
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
        auto meshRead = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto meshLegacy = meshRead.LegacyHostView();
        const auto *mesh = &meshLegacy;
        this->ValidateMeshDependencies();
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateSGS 1");
        DNDS_assert(&uInc != &uIncNew);
        int cnvars = nVars;
        JDiag.GetInvert();
        const index nCellDist = mesh->NumCell();
        sumInc.setZero(cnvars);

#if defined(DNDS_DIST_MT_USE_OMP)
        auto cellOp = [&](WorkerRowRanges worker, TU &localSum)
        {
#else
        const WorkerRowRanges worker{{0, nCellDist}, {}};
        auto &localSum = sumInc;
#endif
            for (index iScan = 0; iScan < worker.Size(); ++iScan)
            {
                index iCell = worker.RowAt(forward ? iScan : worker.Size() - 1 - iScan);

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
                            bool iCellOtherIsThisPart = worker.Contains(iCellOther);
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
                localSum.array() += (uIncNewI - uIncOld).array().abs();

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
        for (int iPart = 0; iPart < this->mesh->NLocalParts(); iPart++)
            cellOp(this->mesh->LocalPartRanges(iPart), sumInc);
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
        auto meshRead = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto meshLegacy = meshRead.LegacyHostView();
        const auto *mesh = &meshLegacy;
        this->ValidateMeshDependencies();
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
        auto meshRequirements = Geom::MeshReadRequirements::CellFlux();
        auto meshRead = mesh->AcquireRead(meshRequirements);
        const auto meshLegacy = meshRead.LegacyHostView();
        const auto *mesh = &meshLegacy;
        this->ValidateMeshDependencies();
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
        for (int iPart = 0; iPart < this->mesh->NLocalParts(); iPart++)
            for (const auto range : this->mesh->LocalPartRanges(iPart).Intervals())
                for (index iScan = range.begin; iScan < range.end; ++iScan)
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
        auto meshRead = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto meshLegacy = meshRead.LegacyHostView();
        const auto *mesh = &meshLegacy;
        this->ValidateMeshDependencies();
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
    void EulerEvaluator<model>::UpdateReactiveSplitChi(
        const ArrayDOFV<nVarsFixed> &u,
        ReactiveSplitDataRefs reactiveSplit,
        real dt,
        OptionalRef<ArrayDOFV<1>> cellTWarm)
    {
        DNDS_check_throw_info(dt > 0, "UpdateReactiveSplitChi requires positive dt");
        DNDS_check_throw_info(Traits::isExtended && settings.reactiveFlow.enabled && phys_.hasChemicalSource(),
                              "UpdateReactiveSplitChi requires reactive extended Euler physics");
        auto &reactiveSplitChi = reactiveSplit.chi;
        auto &reactiveSplitChemicalStep = reactiveSplit.chemicalStep;
        auto &reactiveSplitDiffusiveStep = reactiveSplit.diffusiveStep;
        auto &reactiveSplitShockSensor = reactiveSplit.shockSensor;
        auto &reactiveSplitCoupledScore = reactiveSplit.coupledScore;
        const auto &indicatorSettings = settings.reactiveSplitIndicator;
        const bool useMatchedTransport = indicatorSettings.indicatorMode == ReactiveSplitIndicatorMode::TransportBalance ||
                                         indicatorSettings.indicatorMode == ReactiveSplitIndicatorMode::MatchedRateRatio ||
                                         indicatorSettings.indicatorMode == ReactiveSplitIndicatorMode::ChemicalResponse;
        if (reactiveSplit.stiffnessStep)
            reactiveSplit.stiffnessStep->setConstant(0.0);
        if (indicatorSettings.chiOverride >= 0)
        {
            reactiveSplitChi.setConstant(std::clamp(indicatorSettings.chiOverride, real(0), real(1)));
            reactiveSplitChemicalStep.setConstant(0.0);
            reactiveSplitDiffusiveStep.setConstant(0.0);
            reactiveSplitShockSensor.setConstant(0.0);
            reactiveSplitCoupledScore.setConstant(0.0);
            return;
        }

        DNDS_check_throw_info((indicatorSettings.indicatorMode != ReactiveSplitIndicatorMode::StiffnessRatio && indicatorSettings.indicatorMode != ReactiveSplitIndicatorMode::ChemistryEscape) || reactiveSplit.stiffnessStep,
                              "v2 reactive split indicator requires stiffnessStep storage");

        auto meshRequirements = Geom::MeshReadRequirements::CellFlux();
        auto meshRead = mesh->AcquireRead(meshRequirements);
        const auto meshLegacy = meshRead.LegacyHostView();
        const auto *mesh = &meshLegacy;
        this->ValidateMeshDependencies();
        const int Ns = phys_.nSpecies();
        const int Ns1 = Ns - 1;
        const int Isp = nVars - Ns1;
        const real temperatureFloor = phys_.chem().baseTemperature();
        const index nCellProc = mesh->NumCellProc();
        std::vector<real> temperature(static_cast<size_t>(nCellProc));
        std::vector<real> pressure(static_cast<size_t>(nCellProc));
        std::vector<double> massFractions(static_cast<size_t>(nCellProc) * Ns);

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (index iCell = 0; iCell < nCellProc; ++iCell)
        {
            auto state = u[iCell];
            real TGuess = cellTWarm && iCell < mesh->NumCell() ? (*cellTWarm)[iCell](0) : real(0);
            auto [T, p, asqr, H, gammaEq, gamma] = phys_.conservativeThermal(
                state, TGuess, useMatchedTransport ? real(1e-12) : real(1e-8));
            (void)asqr;
            (void)H;
            (void)gammaEq;
            (void)gamma;
            temperature[static_cast<size_t>(iCell)] = phys_.toPhysT(T);
            pressure[static_cast<size_t>(iCell)] = phys_.toPhysP(p);
            Chemistry::SpeciesBufferView Y{massFractions.data() + static_cast<size_t>(iCell) * Ns, Ns};
            Chemistry::RepairMassFractions(state(0), {state.data() + Isp, Ns1}, Y);
            if (cellTWarm && iCell < mesh->NumCell())
                (*cellTWarm)[iCell](0) = T;
        }

        if (useMatchedTransport)
        {
            DNDS_check_throw_info(phys_.chem().isIdealGas(), "transport balance requires ideal-gas species thermodynamics");
            DNDS_check_throw_info(std::isfinite(settings.reactiveSourceScale) && settings.reactiveSourceScale >= 0,
                                  "transport balance requires finite nonnegative reactiveSourceScale");
            std::vector<double> diffusivity(static_cast<size_t>(nCellProc) * Ns);
            std::vector<double> enthalpy(static_cast<size_t>(nCellProc) * Ns);
            std::vector<real> conductivity(static_cast<size_t>(nCellProc));
            std::vector<real> heatCapacity(static_cast<size_t>(nCellProc));
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(guided)
#endif
            for (index iCell = 0; iCell < nCellProc; ++iCell)
            {
                size_t offset = static_cast<size_t>(iCell) * Ns;
                Chemistry::ConstSpeciesBufferView fractions{massFractions.data() + offset, Ns};
                auto &chem = phys_.chem();
                real temperaturePhysical = temperature[static_cast<size_t>(iCell)];
                real pressurePhysical = pressure[static_cast<size_t>(iCell)];
                chem.speciesDiffusivity(temperaturePhysical, pressurePhysical, fractions, {diffusivity.data() + offset, Ns});
                chem.speciesEnthalpies(temperaturePhysical, pressurePhysical, fractions, {enthalpy.data() + offset, Ns});
                conductivity[static_cast<size_t>(iCell)] = chem.thermalConductivity(temperaturePhysical, pressurePhysical, fractions);
                heatCapacity[static_cast<size_t>(iCell)] = chem.mixtureCv(temperaturePhysical, fractions, pressurePhysical);
            }
            real lengthScale = settings.idealGasProperty.L0;
            real dtPhysical = dt * lengthScale / settings.idealGasProperty.U0;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel
#endif
            {
                std::vector<double> production(static_cast<size_t>(Ns));
                std::vector<real> speciesFlux(static_cast<size_t>(Ns));
                std::vector<real> diffusionSource(static_cast<size_t>(Ns));
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp for schedule(guided)
#endif
                for (index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                {
                    auto &chem = phys_.chem();
                    size_t offset = static_cast<size_t>(iCell) * Ns;
                    real temperaturePhysical = temperature[static_cast<size_t>(iCell)];
                    real densityPhysical = u[iCell](0) * settings.idealGasProperty.rho0;
                    real cv = heatCapacity[static_cast<size_t>(iCell)];
                    Chemistry::ConstSpeciesBufferView fractions{massFractions.data() + offset, Ns};
                    chem.productionRates(temperaturePhysical, pressure[static_cast<size_t>(iCell)], fractions, {production.data(), Ns});
                    std::fill(diffusionSource.begin(), diffusionSource.end(), 0.0);
                    real diffusionEnergy = 0;
                    real shockSensor = 0;
                    auto faces = mesh->cell2face[iCell];
                    for (rowsize faceInCell = 0; faceInCell < faces.size(); ++faceInCell)
                    {
                        index face = faces[faceInCell];
                        index neighbor = mesh->CellFaceOther(iCell, face, faceInCell);
                        if (neighbor == UnInitIndex)
                            continue;
                        rowsize side = mesh->CellIsFaceBack(iCell, face, faceInCell) ? 0 : 1;
                        Geom::tPoint separation = (vfv->GetOtherCellBaryFromCell(iCell, neighbor, face, side) -
                                                   vfv->GetCellBary(iCell)) *
                                                  lengthScale;
                        Geom::tPoint normal = vfv->GetFaceNormFromCell(face, iCell, side, -1) * (side ? -1 : 1);
                        real distanceSquared = separation.squaredNorm();
                        DNDS_check_throw_info(distanceSquared > 0, "transport balance requires distinct neighboring cell centers");
                        real gradientFactor = normal.dot(separation) / distanceSquared;
                        real areaOverVolume = vfv->GetFaceArea(face) / (vfv->GetCellVol(iCell) * lengthScale);
                        size_t neighborOffset = static_cast<size_t>(neighbor) * Ns;
                        real neighborTemperature = temperature[static_cast<size_t>(neighbor)];
                        real neighborPressure = pressure[static_cast<size_t>(neighbor)];
                        real faceDensity = 0.5 * (u[iCell](0) + u[neighbor](0)) * settings.idealGasProperty.rho0;
                        real fluxSum = 0;
                        for (int species = 0; species < Ns; ++species)
                        {
                            real faceDiffusivity = 0.5 * (diffusivity[offset + species] + diffusivity[neighborOffset + species]);
                            speciesFlux[species] = -faceDensity * faceDiffusivity * gradientFactor *
                                                   (massFractions[neighborOffset + species] - massFractions[offset + species]);
                            fluxSum += speciesFlux[species];
                        }
                        real energyFlux = -0.5 * (conductivity[static_cast<size_t>(iCell)] + conductivity[static_cast<size_t>(neighbor)]) *
                                          (neighborTemperature - temperaturePhysical) * gradientFactor;
                        for (int species = 0; species < Ns; ++species)
                        {
                            real faceFraction = 0.5 * (massFractions[offset + species] + massFractions[neighborOffset + species]);
                            real flux = speciesFlux[species] - faceFraction * fluxSum;
                            diffusionSource[species] -= flux * areaOverVolume;
                            energyFlux += 0.5 * (enthalpy[offset + species] + enthalpy[neighborOffset + species]) * flux;
                        }
                        diffusionEnergy -= energyFlux * areaOverVolume;
                        real cellPressure = pressure[static_cast<size_t>(iCell)];
                        shockSensor = std::max(shockSensor, std::abs(neighborPressure - cellPressure) /
                                                                std::max({std::abs(neighborPressure), std::abs(cellPressure), real(1e-30)}));
                    }
                    real chemicalEnergy = 0;
                    real chemicalNormSquared = 0;
                    real diffusionNormSquared = 0;
                    for (int species = 0; species < Ns; ++species)
                    {
                        real internalEnergy = enthalpy[offset + species] - chem.speciesGasConstants()[species] * temperaturePhysical;
                        real massSource = production[species] * chem.molecularWeights()[species] * settings.reactiveSourceScale;
                        chemicalEnergy -= internalEnergy * massSource;
                        diffusionEnergy -= internalEnergy * diffusionSource[species];
                        chemicalNormSquared += sqr(massSource / densityPhysical);
                        diffusionNormSquared += sqr(diffusionSource[species] / densityPhysical);
                    }
                    real energyScale = densityPhysical * std::max(cv, real(1e-30)) * std::max(temperaturePhysical, temperatureFloor);
                    real chemicalActivity = dtPhysical * std::sqrt(chemicalNormSquared + sqr(chemicalEnergy / energyScale));
                    real diffusionActivity = dtPhysical * std::sqrt(diffusionNormSquared + sqr(diffusionEnergy / energyScale));
                    if (indicatorSettings.indicatorMode == ReactiveSplitIndicatorMode::ChemicalResponse)
                    {
                        for (int species = 0; species < Ns; ++species)
                            diffusionSource[species] /= densityPhysical;
                        double autonomousNorm = 0, forcedNorm = 0;
                        chem.constantVolumeSourceResponseNorms(
                            temperaturePhysical, pressure[static_cast<size_t>(iCell)], fractions,
                            {diffusionSource.data(), Ns}, diffusionEnergy / energyScale,
                            settings.reactiveSourceScale, autonomousNorm, forcedNorm);
                        real coupledFraction = ReactiveSplitChemicalResponseFraction(autonomousNorm, forcedNorm, indicatorSettings);
                        reactiveSplitChemicalStep[iCell](0) = sqr(dtPhysical) * autonomousNorm;
                        reactiveSplitDiffusiveStep[iCell](0) = sqr(dtPhysical) * forcedNorm;
                        reactiveSplitShockSensor[iCell](0) = shockSensor;
                        reactiveSplitCoupledScore[iCell](0) = coupledFraction;
                        reactiveSplitChi[iCell](0) = ReactiveSplitSnapChi(1.0 - coupledFraction, indicatorSettings);
                        continue;
                    }
                    real score = indicatorSettings.indicatorMode == ReactiveSplitIndicatorMode::MatchedRateRatio
                                     ? ReactiveSplitMatchedRatioFraction(chemicalActivity, diffusionActivity, shockSensor, indicatorSettings)
                                     : ReactiveSplitBalanceScore(chemicalActivity, diffusionActivity, shockSensor, indicatorSettings);
                    reactiveSplitChemicalStep[iCell](0) = chemicalActivity;
                    reactiveSplitDiffusiveStep[iCell](0) = diffusionActivity;
                    reactiveSplitShockSensor[iCell](0) = shockSensor;
                    reactiveSplitCoupledScore[iCell](0) = score;
                    reactiveSplitChi[iCell](0) = indicatorSettings.indicatorMode == ReactiveSplitIndicatorMode::MatchedRateRatio
                                                     ? ReactiveSplitSnapChi(1.0 - score, indicatorSettings)
                                                 : indicatorSettings.balanceExpansionPasses + indicatorSettings.balanceSmoothingPasses > 0
                                                     ? 1.0 - ReactiveSplitCoupledFraction(score, indicatorSettings)
                                                     : ReactiveSplitChi(score, indicatorSettings);
                }
            }
            const int expansionPasses = indicatorSettings.balanceExpansionPasses;
            const int smoothingPasses = indicatorSettings.balanceSmoothingPasses;
            std::vector<real> nextChi(static_cast<size_t>(mesh->NumCell()));
            for (int pass = 0; pass < expansionPasses + smoothingPasses; ++pass)
            {
                reactiveSplitChi.trans.startPersistentPull();
                reactiveSplitChi.trans.waitPersistentPull();
                for (index cell = 0; cell < mesh->NumCell(); ++cell)
                {
                    real localFraction = 1.0 - reactiveSplitChi[cell](0);
                    real neighborSum = 0;
                    real maximumFraction = localFraction;
                    int neighborCount = 0;
                    auto faces = mesh->cell2face[cell];
                    for (rowsize faceInCell = 0; faceInCell < faces.size(); ++faceInCell)
                    {
                        index neighbor = mesh->CellFaceOther(cell, faces[faceInCell], faceInCell);
                        if (neighbor == UnInitIndex || neighbor == cell)
                            continue;
                        real fraction = 1.0 - reactiveSplitChi[neighbor](0);
                        neighborSum += fraction;
                        maximumFraction = std::max(maximumFraction, fraction);
                        ++neighborCount;
                    }
                    DNDS_check_throw_info(neighborCount <= 2,
                                          "balance spatial filtering requires a one-dimensional cell chain");
                    real fraction = pass < expansionPasses
                                        ? maximumFraction
                                        : ReactiveSplitBalanceSmooth(localFraction, neighborSum, neighborCount);
                    nextChi[static_cast<size_t>(cell)] = 1.0 - fraction;
                }
                for (index cell = 0; cell < mesh->NumCell(); ++cell)
                    reactiveSplitChi[cell](0) = nextChi[static_cast<size_t>(cell)];
            }
            if (expansionPasses + smoothingPasses > 0)
                for (index cell = 0; cell < mesh->NumCell(); ++cell)
                    reactiveSplitChi[cell](0) = ReactiveSplitSnapChi(reactiveSplitChi[cell](0), indicatorSettings);
            return;
        }

        const bool useStiffnessRatio = indicatorSettings.indicatorMode == ReactiveSplitIndicatorMode::StiffnessRatio || indicatorSettings.indicatorMode == ReactiveSplitIndicatorMode::ChemistryEscape;
        std::vector<real> chemicalRate(static_cast<size_t>(mesh->NumCell()), 0.0);
        std::vector<real> maxDiffusivity(static_cast<size_t>(mesh->NumCell()), 0.0);
        std::vector<real> maxStiffnessRate(static_cast<size_t>(mesh->NumCell()), 0.0);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel
#endif
        {
            std::vector<double> omega(static_cast<size_t>(Ns));
            std::vector<double> diffusivity(static_cast<size_t>(Ns));
            std::vector<double> enthalpy(static_cast<size_t>(Ns));
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp for schedule(guided)
#endif
            for (index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            {
                auto &chem = phys_.chem();
                Chemistry::ConstSpeciesBufferView Y{massFractions.data() + static_cast<size_t>(iCell) * Ns, Ns};
                Chemistry::SpeciesBufferView omegaView{omega.data(), Ns};
                Chemistry::SpeciesBufferView diffusivityView{diffusivity.data(), Ns};
                Chemistry::SpeciesBufferView enthalpyView{enthalpy.data(), Ns};
                real T = temperature[static_cast<size_t>(iCell)];
                real p = pressure[static_cast<size_t>(iCell)];
                chem.productionRates(T, p, Y, omegaView);
                chem.speciesDiffusivity(T, p, Y, diffusivityView);
                chem.speciesEnthalpies(T, p, Y, enthalpyView);
                real rhoPhysical = u[iCell](0) * settings.idealGasProperty.rho0;
                real speciesRateSquared = 0;
                real heatRelease = 0;
                for (int k = 0; k < Ns; ++k)
                {
                    real massRate = omega[static_cast<size_t>(k)] * chem.molecularWeights()[static_cast<size_t>(k)];
                    speciesRateSquared += sqr(massRate / rhoPhysical);
                    heatRelease -= massRate * enthalpy[static_cast<size_t>(k)];
                }
                real cv = chem.mixtureCv(T, Y, p);
                real temperatureRateScale = std::abs(heatRelease) /
                                            (rhoPhysical * std::max(cv, real(1e-30)) * std::max(T, temperatureFloor));
                chemicalRate[static_cast<size_t>(iCell)] = std::sqrt(speciesRateSquared + sqr(temperatureRateScale));
                maxDiffusivity[static_cast<size_t>(iCell)] =
                    *std::max_element(diffusivity.begin(), diffusivity.end());
                if (useStiffnessRatio)
                    maxStiffnessRate[static_cast<size_t>(iCell)] =
                        chem.maxChemicalStiffnessRate(T, p, Y);
            }
        }

        real dtPhysical = dt * settings.idealGasProperty.L0 / settings.idealGasProperty.U0;
        std::vector<real> localShockSensor;
        if (indicatorSettings.spatialPasses > 0)
            localShockSensor.resize(static_cast<size_t>(mesh->NumCell()));
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
        for (index iCell = 0; iCell < mesh->NumCell(); ++iCell)
        {
            real maxNormalizedGradient = 0;
            real shockSensor = 0;
            auto c2f = mesh->cell2face[iCell];
            for (rowsize ic2f = 0; ic2f < c2f.size(); ++ic2f)
            {
                index iFace = c2f[ic2f];
                index iCellOther = mesh->CellFaceOther(iCell, iFace, ic2f);
                if (iCellOther == UnInitIndex)
                    continue;
                rowsize if2c = mesh->CellIsFaceBack(iCell, iFace, ic2f) ? 0 : 1;
                real distance = (vfv->GetOtherCellBaryFromCell(iCell, iCellOther, iFace, if2c) -
                                 vfv->GetCellBary(iCell))
                                    .norm();
                if (!(distance > 0))
                    continue;
                real Ti = temperature[static_cast<size_t>(iCell)];
                real Tj = temperature[static_cast<size_t>(iCellOther)];
                real normalizedGradientSquared = sqr((Tj - Ti) / (distance * std::max({Ti, Tj, temperatureFloor})));
                for (int k = 0; k < Ns; ++k)
                {
                    real Yi = massFractions[static_cast<size_t>(iCell) * Ns + k];
                    real Yj = massFractions[static_cast<size_t>(iCellOther) * Ns + k];
                    real speciesScale = std::max({Yi, Yj, real(1e-3)});
                    if (std::max(Yi, Yj) >= 1e-3)
                        normalizedGradientSquared += sqr((Yj - Yi) / (distance * speciesScale));
                }
                maxNormalizedGradient = std::max(maxNormalizedGradient, std::sqrt(normalizedGradientSquared));
                real pi = pressure[static_cast<size_t>(iCell)];
                real pj = pressure[static_cast<size_t>(iCellOther)];
                shockSensor = std::max(shockSensor, std::abs(pj - pi) / std::max({std::abs(pi), std::abs(pj), real(1e-30)}));
            }
            real cellLength = vfv->GetCellMaxLenScale(iCell) * settings.idealGasProperty.L0;
            real gradientLength = maxNormalizedGradient > 0
                                      ? std::max(1.0 / maxNormalizedGradient * settings.idealGasProperty.L0, cellLength)
                                      : veryLargeReal;
            real chemicalActivity = dtPhysical * chemicalRate[static_cast<size_t>(iCell)];
            real diffusionActivity = dtPhysical * maxDiffusivity[static_cast<size_t>(iCell)] / sqr(gradientLength);
            real coupledScore = ReactiveSplitCoupledScore(
                chemicalActivity, diffusionActivity, shockSensor, indicatorSettings);
            reactiveSplitChemicalStep[iCell](0) = chemicalActivity;   // a_i
            reactiveSplitDiffusiveStep[iCell](0) = diffusionActivity; // b_i
            reactiveSplitShockSensor[iCell](0) = shockSensor;         // h_i
            reactiveSplitCoupledScore[iCell](0) = coupledScore;       // C_i
            if (useStiffnessRatio)
            {
                real chemicalStiffness = dtPhysical * maxStiffnessRate[static_cast<size_t>(iCell)]; // s_i
                (*reactiveSplit.stiffnessStep)[iCell](0) = chemicalStiffness;
                if (indicatorSettings.indicatorMode == ReactiveSplitIndicatorMode::ChemistryEscape)
                    reactiveSplitChi[iCell](0) = ReactiveSplitChiChemistryEscape(coupledScore, diffusionActivity, chemicalStiffness, indicatorSettings);
                else
                {
                    real chiLocal = ReactiveSplitChiStiffnessRatio(diffusionActivity, chemicalStiffness, indicatorSettings);
                    // shock gate: never couple across a resolved shock (push toward Strang)
                    real shockFloor = ReactiveSplitSnapChi(1.0 - ReactiveSplitShockGate(shockSensor, indicatorSettings),
                                                           indicatorSettings);
                    reactiveSplitChi[iCell](0) = std::max(chiLocal, shockFloor);
                }
            }
            else
                reactiveSplitChi[iCell](0) = ReactiveSplitChi(coupledScore, indicatorSettings);
            if (indicatorSettings.spatialPasses > 0)
                localShockSensor[static_cast<size_t>(iCell)] = shockSensor;
        }
        for (int iPass = 0; iPass < indicatorSettings.spatialPasses; ++iPass)
        {
            reactiveSplitChi.trans.startPersistentPull();
            reactiveSplitChi.trans.waitPersistentPull();
            std::vector<real> nextChi(static_cast<size_t>(mesh->NumCell()));
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            {
                real neighborCoupledFraction = 0;
                auto c2f = mesh->cell2face[iCell];
                for (rowsize ic2f = 0; ic2f < c2f.size(); ++ic2f)
                {
                    index iFace = c2f[ic2f];
                    index iCellOther = mesh->CellFaceOther(iCell, iFace, ic2f);
                    if (iCellOther == UnInitIndex)
                        continue;
                    neighborCoupledFraction =
                        std::max(neighborCoupledFraction, 1.0 - reactiveSplitChi[iCellOther](0));
                }
                real localCoupledFraction = 1.0 - reactiveSplitChi[iCell](0);
                real expandedCoupledFraction = ReactiveSplitExpandedCoupledFraction(
                    localCoupledFraction, neighborCoupledFraction,
                    localShockSensor[static_cast<size_t>(iCell)], indicatorSettings);
                nextChi[static_cast<size_t>(iCell)] =
                    ReactiveSplitSnapChi(1.0 - expandedCoupledFraction, indicatorSettings);
            }
            for (index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                reactiveSplitChi[iCell](0) = nextChi[static_cast<size_t>(iCell)];
        }
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>, )
    void EulerEvaluator<model>::ReactiveSourceConstVolumeStep(
        ArrayDOFV<nVarsFixed> &u,
        ArrayRECV<nVarsFixed> &uRec,
        real dt,
        real t,
        OptionalRef<ArrayDOFV<1>> cellTWarm,
        OptionalRef<const ArrayDOFV<1>> reactiveSplitChi)
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
            const real splitScale = reactiveSplitChi ? (*reactiveSplitChi)[iCell](0) : real(1);
            DNDS_check_throw_info(std::isfinite(splitScale) && splitScale >= 0 && splitScale <= 1,
                                  fmt::format("ReactiveSourceConstVolumeStep invalid chi at cell {}: {}",
                                              iCell, splitScale));
            if (!(splitScale > 0))
                continue;

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
                settings.reactiveSourceScale * splitScale,
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
    void EulerEvaluator<model>::BuildLimitedVRO2(
        ArrayDOFV<nVarsFixed> &u,
        ArrayRECV<nVarsFixed> &uRecO2,
        ArrayRECV<nVarsFixed> &uRecLimiterWork,
        ArrayRECV<nVarsFixed> &uRecLimiterBuffer,
        CFV::tScalarPair &limiterIndicator,
        ArrayDOFV<1> &o2Beta,
        ArrayDOFV<1> &alpha,
        ArrayDOFV<4> &branchAlpha,
        ArrayDOFV<1> &densityJump,
        ArrayDOFV<1> &pressureJump,
        ArrayDOFV<1> &compression,
        const LimitedVRSettings &limitedVRSettings,
        bool refreshWeakAlpha,
        bool resetWeakAlpha,
        const typename TVFV::template TFBoundary<nVarsFixed> &FBoundary,
        CFV::ReconstructionMaskView reconstructionMask, uint64_t flags)
    {
        auto meshRead = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto meshLegacy = meshRead.LegacyHostView();
        const auto *mesh = &meshLegacy;
        this->ValidateMeshDependencies();
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        static constexpr int iShockAlpha = 0;
        static constexpr int iContactAlpha = 1;
        static constexpr int iContactSupportAlpha = 2;
        static constexpr int iWeakAlpha = 3;
        const bool reuseUnlimited = flags & LVR_Already_Have_UGradBufNoLim;
        const bool reuseLimited = flags & LVR_Already_Have_UGradBuf;
        DNDS_assert(!reuseLimited || reuseUnlimited);

        if (limitedVRSettings.alphaMax == 0 && limitedVRSettings.contactAlphaMax == 0 &&
            limitedVRSettings.weakAlphaMax == 0)
        {
            // Exact unlimited-VR control case: no gradients, PP or exchanges.
            uRecO2.setConstant(0.0);
            alpha.setConstant(0.0);
            branchAlpha.setConstant(0.0);
            densityJump.setConstant(0.0);
            pressureJump.setConstant(0.0);
            compression.setConstant(0.0);
            o2Beta.setConstant(1.0);
            return;
        }

        if (!reuseUnlimited)
            vfv->DoReconstruction2ndGrad(uGradBufNoLim, u, FBoundary, settings.direct2ndRecMethod, reconstructionMask);
        // Only the weak branch reads unrestricted gradients of ghost cells.
        if (!reuseUnlimited && limitedVRSettings.weakAlphaMax > 0 && refreshWeakAlpha)
        {
            uGradBufNoLim.trans.startPersistentPull();
            uGradBufNoLim.trans.waitPersistentPull();
        }

        if (limitedVRSettings.o2ReferenceLimiter == O2ReferenceLimiter::Barth)
        {
            if (!reuseLimited)
                LimiterUGrad(u, uGradBufNoLim, uGradBuf, LIMITER_UGRAD_No_Flags);
            vfv->ConvertUGradToURec(uRecO2, uGradBuf, {}, reconstructionMask);
        }
        else
        {
            DNDS_assert(limitedVRSettings.o2ReferenceLimiter == O2ReferenceLimiter::WBAP);
            vfv->ConvertUGradToURec(uRecO2, uGradBufNoLim, {}, reconstructionMask);
            uRecO2.trans.startPersistentPull();
            uRecO2.trans.waitPersistentPull();

            using tLimitBatch = typename TVFV::template tLimitBatch<nVarsFixed>;
            auto fML = [&](const TU &UL, const TU &UR, const Geom::tPoint &n,
                           const Eigen::Ref<tLimitBatch> &data) -> tLimitBatch
            {
                TU UMean = (UL + UR) * 0.5;
                Eigen::Matrix<real, dim, dim> normBase = Geom::NormBuildLocalBaseV<dim>(n(Seq012));
                UMean(Seq123) = normBase.transpose() * UMean(Seq123);
                Eigen::Vector<real, I4 + 1> UC = UMean(Seq01234);
                real T = phys_.temperature(UMean);
                real gammaEq = phys_.gammaEq(T, UMean);
                real gamma = phys_.gamma(T, UMean);
                auto M = Gas::IdealGas_EulerGasLeftEigenVector<dim>(
                    UC, gammaEq, gamma, phys_.mixtureBaseInternalRhoE(UMean));
                M(EigenAll, Seq123) *= normBase.transpose();
                Eigen::Matrix<real, nVarsFixed, nVarsFixed> ret(nVars, nVars);
                ret.setIdentity();
                ret(Seq01234, Seq01234) = M;
                return (ret * data.transpose()).transpose();
            };
            auto fMR = [&](const TU &UL, const TU &UR, const Geom::tPoint &n,
                           const Eigen::Ref<tLimitBatch> &data) -> tLimitBatch
            {
                TU UMean = (UL + UR) * 0.5;
                Eigen::Matrix<real, dim, dim> normBase = Geom::NormBuildLocalBaseV<dim>(n(Seq012));
                UMean(Seq123) = normBase.transpose() * UMean(Seq123);
                Eigen::Vector<real, I4 + 1> UC = UMean(Seq01234);
                real T = phys_.temperature(UMean);
                real gammaEq = phys_.gammaEq(T, UMean);
                real gamma = phys_.gamma(T, UMean);
                auto M = Gas::IdealGas_EulerGasRightEigenVector<dim>(
                    UC, gammaEq, gamma, phys_.mixtureBaseInternalRhoE(UMean));
                M(Seq123, EigenAll) = normBase * M(Seq123, EigenAll);
                Eigen::Matrix<real, nVarsFixed, nVarsFixed> ret(nVars, nVars);
                ret.setIdentity();
                ret(Seq01234, Seq01234) = M;
                return (ret * data.transpose()).transpose();
            };
            vfv->template DoLimiterWBAP_3<nVarsFixed>(
                u, uRecO2, uRecLimiterWork, uRecLimiterBuffer,
                limiterIndicator, true, fML, fMR, false, 1, reconstructionMask);
        }

        if (limitedVRSettings.o2ReferenceUsePP)
        {
            index nLimited = 0;
            real minimumBeta = 1;
            EvaluateURecBeta(
                u, uRecO2, o2Beta, nLimited, minimumBeta,
                EvaluateURecBeta_COMPRESS_TO_MEAN | EvaluateURecBeta_LOCAL_STATS);
            uRecO2.trans.startPersistentPull();
            uRecO2.trans.waitPersistentPull();
        }
        else
        {
            o2Beta.setConstant(1.0);
            if (limitedVRSettings.o2ReferenceLimiter == O2ReferenceLimiter::Barth)
            {
                uRecO2.trans.startPersistentPull();
                uRecO2.trans.waitPersistentPull();
            }
        }

        std::vector<real> shockAlphaPrevious;
        if (limitedVRSettings.alphaUpdateRelaxation < 1.0 ||
            (limitedVRSettings.alphaMax > 0 &&
             limitedVRSettings.gateMode == LimitedVRGateMode::PressureExtremum))
        {
            shockAlphaPrevious.resize(static_cast<size_t>(mesh->NumCell()));
            for (index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                shockAlphaPrevious[static_cast<size_t>(iCell)] = branchAlpha[iCell](iShockAlpha);
        }
        std::vector<real> contactAlphaPrevious;
        if (limitedVRSettings.alphaUpdateRelaxation < 1.0 ||
            limitedVRSettings.contactUseExtremumCorrection)
        {
            contactAlphaPrevious.resize(static_cast<size_t>(mesh->NumCell()));
            for (index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                contactAlphaPrevious[static_cast<size_t>(iCell)] = branchAlpha[iCell](iContactAlpha);
        }
        std::vector<real> weakAlphaPrevious;
        if (limitedVRSettings.weakAlphaMax > 0 &&
            (!refreshWeakAlpha || limitedVRSettings.weakAlphaUpdateRelaxation < 1.0 ||
             limitedVRSettings.weakAlphaDecayRetention > 0))
        {
            weakAlphaPrevious.resize(static_cast<size_t>(mesh->NumCell()));
            for (index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                weakAlphaPrevious[static_cast<size_t>(iCell)] = branchAlpha[iCell](iWeakAlpha);
        }
        alpha.setConstant(0.0);
        branchAlpha.setConstant(0.0);
        densityJump.setConstant(0.0);
        pressureJump.setConstant(0.0);
        compression.setConstant(0.0);
        std::vector<real> weakPressureMean;
        std::vector<TVec> weakPressureGradient;
        if (limitedVRSettings.weakAlphaMax > 0 && refreshWeakAlpha)
        {
            weakPressureMean.resize(static_cast<size_t>(mesh->NumCellProc()));
            weakPressureGradient.resize(static_cast<size_t>(mesh->NumCellProc()));
            auto eBaseSpecies = phys_.mixtureBaseInternalRhoESpecies();
            for (index iCell = 0; iCell < mesh->NumCellProc(); ++iCell)
            {
                auto [T, p, asqr, H, gammaEq, gamma] = phys_.conservativeThermal(u[iCell]);
                TDiffU gradPrimitive;
                TDiffU gradConservative(dim, nVars);
                gradConservative.setZero();
                gradConservative.topRows(gDim) = uGradBufNoLim[iCell];
                Gas::GradientCons2Prim_IdealGas<dim>(
                    u[iCell], gradConservative, gradPrimitive, gammaEq, eBaseSpecies);
                weakPressureMean[static_cast<size_t>(iCell)] = p;
                weakPressureGradient[static_cast<size_t>(iCell)] = gradPrimitive(Seq012, I4);
            }
        }
        for (index iFace = 0; iFace < mesh->NumFaceProc(); ++iFace)
        {
            auto f2c = mesh->face2cell[iFace];
            if (f2c[1] == UnInitIndex)
                continue;

            TU UL = u[f2c[0]];
            UL += (vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, -1, std::array<int, 1>{0}, 1) *
                   uRecO2[f2c[0]])
                      .transpose();
            UFromCell2Face(UL, iFace, f2c[0], 0);

            TU UR = u[f2c[1]];
            UR += (vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, -1, std::array<int, 1>{0}, 1) *
                   uRecO2[f2c[1]])
                      .transpose();
            UFromCell2Face(UR, iFace, f2c[1], 1);

            auto [TL, pL, asqrL, HL, gammaEqL, gammaL] = phys_.conservativeThermal(UL);
            auto [TR, pR, asqrR, HR, gammaEqR, gammaR] = phys_.conservativeThermal(UR);
            TVec unitNorm = vfv->GetFaceNorm(iFace, -1)(Seq012);
            TVec velocityL = UL(Seq123) / UL(0);
            TVec velocityR = UR(Seq123) / UR(0);
            auto sensor = EvaluateLimitedVRFaceSensor(
                pL, pR, velocityL.dot(unitNorm), velocityR.dot(unitNorm),
                std::sqrt(std::max(asqrL, real(0))), std::sqrt(std::max(asqrR, real(0))),
                limitedVRSettings);
            real contactDensityLeft = limitedVRSettings.contactUseCellMeanJump ? u[f2c[0]](0) : UL(0);
            real contactDensityRight = limitedVRSettings.contactUseCellMeanJump ? u[f2c[1]](0) : UR(0);
            auto contactSensor = EvaluateLimitedVRContactFaceSensor(
                contactDensityLeft, contactDensityRight, limitedVRSettings);
            auto contactSupportSensor = contactSensor;
            real weakAlpha = 0;
            if (limitedVRSettings.weakAlphaMax > 0 && refreshWeakAlpha)
            {
                TU meanL = u[f2c[0]], meanR = u[f2c[1]];
                UFromCell2Face(meanL, iFace, f2c[0], 0);
                UFromCell2Face(meanR, iFace, f2c[1], 1);
                auto [TmL, pmL, amsqrL, HmL, gmEqL, gmL] = phys_.conservativeThermal(meanL);
                auto [TmR, pmR, amsqrR, HmR, gmEqR, gmR] = phys_.conservativeThermal(meanR);
                TVec dxL = (vfv->GetFaceQuadraturePPhysFromCell(iFace, f2c[0], 0, -1) -
                            vfv->GetCellQuadraturePPhys(f2c[0], -1))(Seq012);
                TVec dxR = (vfv->GetFaceQuadraturePPhysFromCell(iFace, f2c[1], 1, -1) -
                            vfv->GetCellQuadraturePPhys(f2c[1], -1))(Seq012);
                weakAlpha = EvaluateLimitedVRWeakAlpha(
                    weakPressureMean[static_cast<size_t>(f2c[0])],
                    weakPressureMean[static_cast<size_t>(f2c[1])],
                    weakPressureGradient[static_cast<size_t>(f2c[0])].dot(dxL),
                    weakPressureGradient[static_cast<size_t>(f2c[1])].dot(dxR),
                    (meanL(Seq123) / meanL(0)).dot(unitNorm),
                    (meanR(Seq123) / meanR(0)).dot(unitNorm),
                    0.5 * (meanL(0) * std::sqrt(std::max(amsqrL, real(0))) +
                           meanR(0) * std::sqrt(std::max(amsqrR, real(0)))),
                    limitedVRSettings);
            }
            if (limitedVRSettings.contactExtremumUseCellMeanSupport &&
                !limitedVRSettings.contactUseCellMeanJump)
                contactSupportSensor = EvaluateLimitedVRContactFaceSensor(
                    u[f2c[0]](0), u[f2c[1]](0), limitedVRSettings);

            for (int if2c = 0; if2c < 2; ++if2c)
            {
                index iCell = f2c[if2c];
                if (iCell >= mesh->NumCell())
                    continue;
                branchAlpha[iCell](iShockAlpha) =
                    std::max(branchAlpha[iCell](iShockAlpha), sensor.alpha);
                branchAlpha[iCell](iContactAlpha) =
                    std::max(branchAlpha[iCell](iContactAlpha), contactSensor.alpha);
                branchAlpha[iCell](iContactSupportAlpha) =
                    std::max(branchAlpha[iCell](iContactSupportAlpha), contactSupportSensor.alpha);
                branchAlpha[iCell](iWeakAlpha) = std::max(branchAlpha[iCell](iWeakAlpha), weakAlpha);
                densityJump[iCell](0) = std::max(densityJump[iCell](0), contactSensor.densityJump);
                pressureJump[iCell](0) = std::max(pressureJump[iCell](0), sensor.pressureJump);
                compression[iCell](0) = std::max(compression[iCell](0), sensor.compression);
            }
        }

        const bool correctPressure = limitedVRSettings.alphaMax > 0 &&
                                     limitedVRSettings.gateMode == LimitedVRGateMode::PressureExtremum;
        const bool correctDensity = limitedVRSettings.contactUseExtremumCorrection &&
                                    limitedVRSettings.contactAlphaMax > 0;
        const index nOwned = mesh->NumCell();
        std::vector<bool> correctionBoundary;
        if (correctPressure || correctDensity)
            correctionBoundary.resize(static_cast<size_t>(nOwned), false);
        std::vector<real> shockAlphaBase, contactAlphaBase;

        // Each component has its own stencil/depth, but all active components
        // use one packed pull per round. Do not use the O2 or PP arrays as scratch.
        std::vector<std::array<real, 3>> expandedGates;
        auto expandGates = [&](bool shock, bool contact, bool weak)
        {
            expandedGates.resize(static_cast<size_t>(nOwned));
            for (index iCell = 0; iCell < nOwned; ++iCell)
                expandedGates[static_cast<size_t>(iCell)] = {
                    branchAlpha[iCell](iShockAlpha), branchAlpha[iCell](iContactAlpha),
                    branchAlpha[iCell](iWeakAlpha)};
            for (index iFace = 0; iFace < mesh->NumFaceProc(); ++iFace)
            {
                auto f2c = mesh->face2cell[iFace];
                if (f2c[1] == UnInitIndex)
                    continue;
                for (int side = 0; side < 2; ++side)
                {
                    index iCell = f2c[side], iOther = f2c[1 - side];
                    if (iCell >= nOwned)
                        continue;
                    const size_t i = static_cast<size_t>(iCell);
                    auto &expanded = expandedGates[i];
                    if (shock && !correctionBoundary[i])
                        expanded[0] = std::max(expanded[0], branchAlpha[iOther](iShockAlpha));
                    if (contact && !correctionBoundary[i])
                        expanded[1] = std::max(expanded[1], branchAlpha[iOther](iContactAlpha));
                    if (weak)
                        expanded[2] = std::max(expanded[2], branchAlpha[iOther](iWeakAlpha));
                }
            }
            for (index iCell = 0; iCell < nOwned; ++iCell)
            {
                const auto &expanded = expandedGates[static_cast<size_t>(iCell)];
                if (shock)
                    branchAlpha[iCell](iShockAlpha) = expanded[0];
                if (contact)
                    branchAlpha[iCell](iContactAlpha) = expanded[1];
                if (weak)
                    branchAlpha[iCell](iWeakAlpha) = expanded[2];
            }
        };
        int weakLayersRemaining = limitedVRSettings.weakAlphaMax > 0 && refreshWeakAlpha
                                      ? limitedVRSettings.weakHaloLayers
                                      : 0;
        if (correctPressure || correctDensity)
        {
            // Round 1 supplies both correction supports; the fresh weak gate
            // can already take its first maximum pass from the same payload.
            branchAlpha.trans.startPersistentPull();
            branchAlpha.trans.waitPersistentPull();
            if (weakLayersRemaining > 0)
            {
                expandGates(false, false, true);
                --weakLayersRemaining;
            }
        }

        if (correctPressure)
        {
            // The base face gate remains the local product gate.  Add penalty
            // only at a strict cell-mean pressure extremum next to that detected
            // shock core.  This covers an oscillatory pre/post-shock cell without
            // spreading the gate across an otherwise monotone shock profile.
            const index nCell = mesh->NumCell();
            const index nCellProc = mesh->NumCellProc();
            std::vector<real> pressureMean(static_cast<size_t>(nCellProc));
            for (index iCell = 0; iCell < nCellProc; ++iCell)
            {
                auto [T, p, asqr, H, gammaEq, gamma] = phys_.conservativeThermal(u[iCell]);
                pressureMean[static_cast<size_t>(iCell)] = p;
            }
            std::vector<real> nearestLower(static_cast<size_t>(nCell), -veryLargeReal);
            std::vector<real> nearestUpper(static_cast<size_t>(nCell), veryLargeReal);
            std::vector<real> shockSupport(static_cast<size_t>(nCell), 0.0);
            std::vector<bool> hasLower(static_cast<size_t>(nCell), false);
            std::vector<bool> hasUpper(static_cast<size_t>(nCell), false);
            std::vector<int> equalNeighbourCount(static_cast<size_t>(nCell), 0);
            auto &boundaryAdjacent = correctionBoundary;
            shockAlphaBase.resize(static_cast<size_t>(nCell));
            for (index iCell = 0; iCell < nCell; ++iCell)
                shockAlphaBase[static_cast<size_t>(iCell)] = branchAlpha[iCell](iShockAlpha);
            for (index iFace = 0; iFace < mesh->NumFaceProc(); ++iFace)
            {
                auto f2c = mesh->face2cell[iFace];
                if (f2c[1] == UnInitIndex)
                {
                    if (f2c[0] < nCell)
                        boundaryAdjacent[static_cast<size_t>(f2c[0])] = true;
                    continue;
                }
                for (int if2c = 0; if2c < 2; ++if2c)
                {
                    index iCell = f2c[if2c];
                    index iOther = f2c[1 - if2c];
                    if (iCell >= nCell)
                        continue;
                    const size_t i = static_cast<size_t>(iCell);
                    real pCell = pressureMean[i];
                    real pOther = pressureMean[static_cast<size_t>(iOther)];
                    real pScale = std::max({std::abs(pCell), std::abs(pOther), real(1e-30)});
                    real equalTolerance = limitedVRSettings.oscillationTransverseTolerance * pScale;
                    if (pOther < pCell - equalTolerance)
                    {
                        nearestLower[i] = std::max(nearestLower[i], pOther);
                        hasLower[i] = true;
                    }
                    else if (pOther > pCell + equalTolerance)
                    {
                        nearestUpper[i] = std::min(nearestUpper[i], pOther);
                        hasUpper[i] = true;
                    }
                    else
                        ++equalNeighbourCount[i];
                    shockSupport[i] = std::max(shockSupport[i], branchAlpha[iOther](iShockAlpha));
                }
            }
            for (index iCell = 0; iCell < nCell; ++iCell)
            {
                const size_t i = static_cast<size_t>(iCell);
                branchAlpha[iCell](iShockAlpha) = 0;
                if (boundaryAdjacent[i] ||
                    equalNeighbourCount[i] < limitedVRSettings.oscillationMinTransverseNeighbours)
                    continue;
                real pCell = pressureMean[i];
                real extremum = 0;
                if (hasLower[i] && !hasUpper[i])
                    extremum = (pCell - nearestLower[i]) /
                               std::max({std::abs(pCell), std::abs(nearestLower[i]), real(1e-30)});
                else if (hasUpper[i] && !hasLower[i])
                    extremum = (nearestUpper[i] - pCell) /
                               std::max({std::abs(pCell), std::abs(nearestUpper[i]), real(1e-30)});
                real supportFraction = shockSupport[i] / std::max(limitedVRSettings.alphaMax, real(1e-30));
                real response =
                    LimitedVRCubicRamp(
                        extremum,
                        limitedVRSettings.oscillationExtremumStart,
                        limitedVRSettings.oscillationExtremumFull) *
                    LimitedVRCubicRamp(
                        supportFraction,
                        limitedVRSettings.oscillationSupportStart,
                        limitedVRSettings.oscillationSupportFull);
                real retainedResponse =
                    limitedVRSettings.oscillationRetention *
                    shockAlphaPrevious[i] / std::max(limitedVRSettings.alphaMax, real(1e-30)) *
                    LimitedVRCubicRamp(
                        supportFraction,
                        limitedVRSettings.oscillationSupportStart,
                        limitedVRSettings.oscillationSupportFull);
                branchAlpha[iCell](iShockAlpha) = limitedVRSettings.alphaMax * std::max(response, retainedResponse);
            }
            // Keep the seed separate from its base gate until its own one-pass
            // correction expansion is complete; expanding the base would alter v2.
        }

        if (correctDensity)
        {
            // Mirror the supported pressure-extremum correction on cell-mean density.  The
            // separately packed contact-support field may use cell-mean jumps
            // without applying those jumps as the base contact penalty.  Only
            // a strict local density extremum with the configured transverse
            // equal-state neighbours can add penalty.
            const index nCell = mesh->NumCell();
            const index nCellProc = mesh->NumCellProc();
            std::vector<real> densityMean(static_cast<size_t>(nCellProc));
            for (index iCell = 0; iCell < nCellProc; ++iCell)
                densityMean[static_cast<size_t>(iCell)] = u[iCell](0);
            std::vector<real> nearestLower(static_cast<size_t>(nCell), -veryLargeReal);
            std::vector<real> nearestUpper(static_cast<size_t>(nCell), veryLargeReal);
            std::vector<real> contactSupport(static_cast<size_t>(nCell), 0.0);
            std::vector<bool> hasLower(static_cast<size_t>(nCell), false);
            std::vector<bool> hasUpper(static_cast<size_t>(nCell), false);
            std::vector<int> equalNeighbourCount(static_cast<size_t>(nCell), 0);
            auto &boundaryAdjacent = correctionBoundary;
            contactAlphaBase.resize(static_cast<size_t>(nCell));
            for (index iCell = 0; iCell < nCell; ++iCell)
                contactAlphaBase[static_cast<size_t>(iCell)] = branchAlpha[iCell](iContactAlpha);
            for (index iFace = 0; iFace < mesh->NumFaceProc(); ++iFace)
            {
                auto f2c = mesh->face2cell[iFace];
                if (f2c[1] == UnInitIndex)
                {
                    if (f2c[0] < nCell)
                        boundaryAdjacent[static_cast<size_t>(f2c[0])] = true;
                    continue;
                }
                for (int if2c = 0; if2c < 2; ++if2c)
                {
                    index iCell = f2c[if2c];
                    index iOther = f2c[1 - if2c];
                    if (iCell >= nCell)
                        continue;
                    const size_t i = static_cast<size_t>(iCell);
                    real densityCell = densityMean[i];
                    real densityOther = densityMean[static_cast<size_t>(iOther)];
                    real densityScale = std::max(
                        {std::abs(densityCell), std::abs(densityOther), real(1e-30)});
                    real equalTolerance =
                        limitedVRSettings.oscillationTransverseTolerance * densityScale;
                    if (densityOther < densityCell - equalTolerance)
                    {
                        nearestLower[i] = std::max(nearestLower[i], densityOther);
                        hasLower[i] = true;
                    }
                    else if (densityOther > densityCell + equalTolerance)
                    {
                        nearestUpper[i] = std::min(nearestUpper[i], densityOther);
                        hasUpper[i] = true;
                    }
                    else
                        ++equalNeighbourCount[i];
                    contactSupport[i] = std::max(
                        contactSupport[i], branchAlpha[iOther](iContactSupportAlpha));
                }
            }
            for (index iCell = 0; iCell < nCell; ++iCell)
            {
                const size_t i = static_cast<size_t>(iCell);
                branchAlpha[iCell](iContactAlpha) = 0;
                if (boundaryAdjacent[i] ||
                    equalNeighbourCount[i] < limitedVRSettings.oscillationMinTransverseNeighbours)
                    continue;
                real densityCell = densityMean[i];
                real extremum = 0;
                if (hasLower[i] && !hasUpper[i])
                    extremum = (densityCell - nearestLower[i]) /
                               std::max({std::abs(densityCell), std::abs(nearestLower[i]), real(1e-30)});
                else if (hasUpper[i] && !hasLower[i])
                    extremum = (nearestUpper[i] - densityCell) /
                               std::max({std::abs(densityCell), std::abs(nearestUpper[i]), real(1e-30)});
                real supportFraction =
                    contactSupport[i] / std::max(limitedVRSettings.contactAlphaMax, real(1e-30));
                real response =
                    LimitedVRCubicRamp(
                        extremum,
                        limitedVRSettings.contactExtremumStart,
                        limitedVRSettings.contactExtremumFull) *
                    LimitedVRCubicRamp(
                        supportFraction,
                        limitedVRSettings.contactSupportStart,
                        limitedVRSettings.contactSupportFull);
                real retainedResponse =
                    limitedVRSettings.contactExtremumRetention *
                    contactAlphaPrevious[i] /
                    std::max(limitedVRSettings.contactAlphaMax, real(1e-30)) *
                    LimitedVRCubicRamp(
                        supportFraction,
                        limitedVRSettings.contactSupportStart,
                        limitedVRSettings.contactSupportFull);
                branchAlpha[iCell](iContactAlpha) =
                    limitedVRSettings.contactAlphaMax * std::max(response, retainedResponse);
            }
        }

        auto mergeBaseGate = [&](int component, const std::vector<real> &base)
        {
            for (index iCell = 0; iCell < nOwned; ++iCell)
                branchAlpha[iCell](component) = std::max(
                    branchAlpha[iCell](component), base[static_cast<size_t>(iCell)]);
        };
        const int pressureLayers = correctPressure ? 1 : 0;
        const int densityLayers = correctDensity ? limitedVRSettings.contactExtremumHaloLayers : 0;
        if (correctDensity && densityLayers == 0)
            mergeBaseGate(iContactAlpha, contactAlphaBase);
        const int nExpansionRounds = std::max({pressureLayers, densityLayers, weakLayersRemaining});
        for (int iLayer = 0; iLayer < nExpansionRounds; ++iLayer)
        {
            // Round 2 packs pressure/density seeds and the once-expanded weak
            // field. For H2, round 3 finishes density; no trailing pull is needed.
            branchAlpha.trans.startPersistentPull();
            branchAlpha.trans.waitPersistentPull();
            expandGates(iLayer < pressureLayers, iLayer < densityLayers,
                        iLayer < weakLayersRemaining);
            if (correctPressure && iLayer + 1 == pressureLayers)
                mergeBaseGate(iShockAlpha, shockAlphaBase);
            if (correctDensity && iLayer + 1 == densityLayers)
                mergeBaseGate(iContactAlpha, contactAlphaBase);
        }
        if (limitedVRSettings.weakAlphaMax > 0 &&
            (!refreshWeakAlpha || limitedVRSettings.weakAlphaUpdateRelaxation < 1.0 ||
             limitedVRSettings.weakAlphaDecayRetention > 0))
            for (index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                branchAlpha[iCell](iWeakAlpha) = UpdateLimitedVRWeakAlpha(
                    weakAlphaPrevious[static_cast<size_t>(iCell)],
                    branchAlpha[iCell](iWeakAlpha),
                    refreshWeakAlpha,
                    resetWeakAlpha,
                    limitedVRSettings.weakAlphaUpdateRelaxation,
                    limitedVRSettings.weakAlphaDecayRetention);

        // A compact face-neighbour halo covers the one-sided cells immediately
        // adjacent to a detected shock.  Propagating the already mapped alpha
        // avoids lowering the local indicator thresholds into smooth wall
        // regions.  Pull before every layer so partition-boundary propagation
        // is identical to an unpartitioned traversal.
        std::vector<real> expandedAlpha;
        if (limitedVRSettings.gateHaloLayers > 0)
            expandedAlpha.resize(static_cast<size_t>(mesh->NumCell()));
        for (int iLayer = 0; iLayer < limitedVRSettings.gateHaloLayers; ++iLayer)
        {
            branchAlpha.trans.startPersistentPull();
            branchAlpha.trans.waitPersistentPull();
            auto &expanded = expandedAlpha;
            for (index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                expanded[static_cast<size_t>(iCell)] = branchAlpha[iCell](iShockAlpha);
            for (index iFace = 0; iFace < mesh->NumFaceProc(); ++iFace)
            {
                auto f2c = mesh->face2cell[iFace];
                if (f2c[1] == UnInitIndex)
                    continue;
                real faceAlpha = std::max(
                    branchAlpha[f2c[0]](iShockAlpha), branchAlpha[f2c[1]](iShockAlpha));
                if (limitedVRSettings.gateHaloSaturatedOnly)
                    faceAlpha = faceAlpha >= limitedVRSettings.alphaMax * (1.0 - 1e-12)
                                    ? limitedVRSettings.alphaMax
                                    : 0.0;
                for (int if2c = 0; if2c < 2; ++if2c)
                    if (f2c[if2c] < mesh->NumCell())
                        expanded[static_cast<size_t>(f2c[if2c])] =
                            std::max(expanded[static_cast<size_t>(f2c[if2c])], faceAlpha);
            }
            for (index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                branchAlpha[iCell](iShockAlpha) = expanded[static_cast<size_t>(iCell)];
        }
        if (limitedVRSettings.alphaUpdateRelaxation < 1.0)
            for (index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            {
                branchAlpha[iCell](iShockAlpha) =
                    limitedVRSettings.alphaUpdateRelaxation * branchAlpha[iCell](iShockAlpha) +
                    (1.0 - limitedVRSettings.alphaUpdateRelaxation) *
                        shockAlphaPrevious[static_cast<size_t>(iCell)];
                branchAlpha[iCell](iContactAlpha) =
                    limitedVRSettings.alphaUpdateRelaxation * branchAlpha[iCell](iContactAlpha) +
                    (1.0 - limitedVRSettings.alphaUpdateRelaxation) *
                        contactAlphaPrevious[static_cast<size_t>(iCell)];
            }
        for (index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            alpha[iCell](0) = std::max(branchAlpha[iCell](iWeakAlpha), CombineLimitedVRAlpha(
                                                                           branchAlpha[iCell](iShockAlpha), branchAlpha[iCell](iContactAlpha)));
    }

    DNDS_SWITCH_INTELLISENSE(
        template <EulerModel model>, template <>)
    void EulerEvaluator<model>::LimiterUGrad(
        ArrayDOFV<nVarsFixed> &u, ArrayGRADV<nVarsFixed, gDim> &uGrad, ArrayGRADV<nVarsFixed, gDim> &uGradNew,
        uint64_t flags)
    {
        auto meshRead = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto meshLegacy = meshRead.LegacyHostView();
        const auto *mesh = &meshLegacy;
        this->ValidateMeshDependencies();
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
     *  Reports global statistics unless LOCAL_STATS requests local diagnostics;
     *  positivity-floor reductions are retained in both modes.
     *
     *  @param u         Conservative variable DOF array.
     *  @param uRec      Reconstruction coefficients (may be modified for order reduction).
     *  RANS never participates in flow base selection, order reduction or beta.
     *  Two-equation scalar coefficients are bounded separately about their means,
     *  with a common scalar factor and the existing 0.9 safety margin. SA follows
     *  USE_NS_SA_NUT_REDUCED_ORDER; otherwise negative SA reconstruction is retained.
     *
     *  @par PP POLICY
     *  The analytic density/thermal beta uses the complete raw flow/species
     *  polynomial and the affine mixture reference, but no species-simplex
     *  bound. Raw reconstructed rhoY is retained for conservative advection and
     *  diffusion. Final physical acceptance evaluates a temporary
     *  clipped/normalized composition, matching the EOS/T input without writing
     *  it back. `fallbackToBase` preserves Euler's all-or-nothing response;
     *  `backtracking` searches an intermediate common beta. RANS is detached.
     *
     *  @param uRecBeta  Per-cell flow/species compression coefficient (output).
     *  @param nLim      Number of cells with flow beta < 1; excludes scalar-only repairs.
     *  @param betaMin   Minimum flow beta; excludes scalar-only repairs.
     *  @param flag      Evaluation mode flags.
     */
    void EulerEvaluator<model>::EvaluateURecBeta(
        ArrayDOFV<nVarsFixed> &u,
        ArrayRECV<nVarsFixed> &uRec,
        ArrayDOFV<1> &uRecBeta, index &nLim, real &betaMin,
        int flag)
    {
        auto meshRead = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto meshLegacy = meshRead.LegacyHostView();
        const auto *mesh = &meshLegacy;
        this->ValidateMeshDependencies();
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
            MPI::AllreduceOneReal(rhoMin, MPI_MIN, this->mesh->getMPI());
            MPI::AllreduceOneReal(rhoEiMin, MPI_MIN, this->mesh->getMPI());
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
            const auto clippedReferenceGood = [&](const Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> &rec)
            {
                for (int iG = 0; iG < rec.rows(); ++iG)
                {
                    const typename EulerEvaluator<model>::TU point = rec.row(iG);
                    if (!(point(0) >= rhoEps) || !point.allFinite())
                        return false;
                    const real kinetic = .5 * point(Seq123).squaredNorm() / point(0);
                    // mixtureBaseInternalRhoE performs only a temporary
                    // clip/normalization. The raw polynomial remains untouched
                    // for the conservative species fluxes.
                    const real sensible = point(I4) - kinetic - phys_.mixtureBaseInternalRhoE(point);
                    if (!(sensible >= rhoeSensibleEps))
                        return false;
                }
                return true;
            };

            // Scalar PP is independent of the flow base, and runs even when flow
            // passes immediately. Scale about the mean, not a lower-order base.
#ifdef USE_NS_SA_NUT_REDUCED_ORDER
            constexpr bool limitSA = true;
#else
            constexpr bool limitSA = false;
#endif
            if constexpr (Traits::has2EQ || (Traits::hasSA && limitSA))
            {
                constexpr int nScalars = Traits::has2EQ ? 2 : 1;
                const auto scalars = Eigen::seqN(I4 + 1, Eigen::fix<nScalars>);
                const auto scalarIncrement = (quadBase * uRec[iCell](EigenAll, scalars)).eval();
                real thetaScalar = 1;
                for (int scalar = 0; scalar < nScalars; ++scalar)
                {
                    const real mean = u[iCell](I4 + 1 + scalar);
                    DNDS_assert_info(std::isfinite(mean) && mean >= 0,
                                     "reconstruction PP requires finite nonnegative RANS means");
                    for (int q = 0; q < scalarIncrement.rows(); ++q)
                        if (mean + scalarIncrement(q, scalar) < 0)
                            thetaScalar = std::min(thetaScalar, mean / (-scalarIncrement(q, scalar) + verySmallReal));
                }
                DNDS_assert(thetaScalar >= 0 && thetaScalar <= 1 && std::isfinite(thetaScalar));
                if (thetaScalar < 1)
                    uRec[iCell](EigenAll, scalars) *= real(0.9) * thetaScalar;
            }

            // Flow/species PP: retain the existing staged base-selection algorithm.
            int curOrder = vfv->GetCellOrder(iCell);
            Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> uRecBase = uRec[iCell];
            Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> recBase; // * has to call checkRecBaseGood() to hold valid value
            auto checkRecBaseGood = [&]() -> bool
            {
                recBase = (quadBase * uRecBase).rowwise() + u[iCell].transpose();
                if (recBase(EigenAll, 0).minCoeff() < rhoEps) // TODO: add relaxation to eps values
                    return false;
                Eigen::Vector<real, Eigen::Dynamic> ek =
                    0.5 * (recBase(EigenAll, Seq123).array().square().rowwise().sum()) / recBase(EigenAll, 0).array();
                Eigen::Vector<real, Eigen::Dynamic> rhoE_base_q = rhoE_base_perQ(recBase);
                Eigen::Vector<real, Eigen::Dynamic> eInternalS = (recBase(EigenAll, I4) - ek - rhoE_base_q);
                if (eInternalS.minCoeff() < rhoeSensibleEps)
                    return false;
                if (!clippedReferenceGood(recBase))
                    return false;
                // Nonconstant polynomial bases are selected by the locked
                // flow/species thermal condition, not by the raw species
                // simplex. Point-property evaluation repairs a temporary
                // composition; forcing another order reduction for a
                // species-only violation needlessly discards flow accuracy.
                // The cell mean simplex was asserted above and remains the
                // constant target when beta later tends to zero.
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
                // Reject this order for flow/species only. RANS coefficients have
                // their own correction above and must survive every failed base.
                if constexpr (Traits::hasSA || Traits::has2EQ)
                    uRec[iCell](EigenAll, Seq01234) = uRecBase(EigenAll, Seq01234);
                else
                    uRec[iCell] = uRecBase;
                curOrder--;
            }

            const Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> uRecTarget = uRec[iCell];
            const Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed>
                recIncOriginal = quadBase * (uRecTarget - uRecBase);
            Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> recInc = recIncOriginal;
            Eigen::Vector<real, Eigen::Dynamic> rhoS = recInc(EigenAll, 0) + recBase(EigenAll, 0);
            Eigen::Index rhoMinIdx{-1};
            real rhoMin = rhoS.minCoeff(&rhoMinIdx);
            real theta1 = 1;
            if (rhoMin < rhoEps)
                for (int iG = 0; iG < rhoS.size(); iG++)
                    if (recInc(iG, 0) < 0) // negative increment
                        theta1 = std::min(theta1, (recBase(iG, 0) - rhoEps) / (-recInc(iG, 0) + verySmallReal));
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

            // TODO(PP-RECONSTRUCTION-CLIPPED-RHOY): evaluate, with matched
            // Euler/Cauchy oracles, whether this analytic thermal beta should
            // use the clipped/normalized temporary-composition reference rather
            // than the raw affine reference. The clipped map is heuristic and
            // non-convex; retain final endpoint checking/backtracking for either
            // choice. See docs/architecture/CauchyPositivity.md.
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

            // PP POLICY -- clipped reconstruction endpoint. This reads a
            // temporary clipped/normalized composition only for the energy/T
            // predicate. uRecTarget remains raw and continues to species
            // advection/diffusion. The compatibility policy drops the complete
            // flow/species increment to its clipped-valid base.
            if (settings.uRecClippedReferencePolicy == "fallbackToBase" &&
                !clippedReferenceGood(recVRhoG))
                thetaP = 0;

            uRecBeta[iCell](0) = theta1 * thetaP;
            // if (uRecBeta[iCell](0) < 1)
            //     uRecBeta[iCell](0) *= 1 - 1e-8;
            // uRecBeta[iCell](0) = std::min(theta1, thetaP);
            if (uRecBeta[iCell](0) < 1)
            {
                // uRecBeta[iCell](0) *= uRecBeta[iCell](0) < 0.99 ? 0. : 0.99; //! for safety
                uRecBeta[iCell](0) *= std::pow(uRecBeta[iCell](0), static_cast<int>(std::round(settings.uRecBetaCompressPower))) * safetyRatio;
                if (uRecBeta[iCell](0) < betaCutOff)
                    uRecBeta[iCell](0) = 0;
            }
            if (settings.uRecClippedReferencePolicy == "backtracking")
            {
                int iteration = 0;
                auto candidate = (recBase + uRecBeta[iCell](0) * recIncOriginal).eval();
                while (!clippedReferenceGood(candidate) && iteration < 1000)
                {
                    uRecBeta[iCell](0) *= .9;
                    if (uRecBeta[iCell](0) < betaCutOff)
                        uRecBeta[iCell](0) = 0;
                    candidate = (recBase + uRecBeta[iCell](0) * recIncOriginal).eval();
                    ++iteration;
                }
                DNDS_assert_info(clippedReferenceGood(candidate),
                                 "clipped-reference reconstruction backtracking did not reach its validated base");
            }
            if (uRecBeta[iCell](0) < 0)
            {
                std::cout << fmt::format("theta1 {}, thetaP {}", theta1, thetaP) << std::endl;
                DNDS_assert(false);
            }
            const auto applyFlowSpeciesBeta = [&](real beta)
            {
                uRec[iCell](EigenAll, Seq01234) = (uRecTarget(EigenAll, Seq01234) - uRecBase(EigenAll, Seq01234)) * beta + uRecBase(EigenAll, Seq01234);
                if constexpr (Traits::isExtended)
                {
                    if (phys_.hasChemicalSource())
                    {
                        int Ns1 = phys_.nSpecies() - 1;
                        int Isp = nVars - Ns1;
                        auto seqSpecies = Eigen::seq(Isp, EigenLast);
                        uRec[iCell](EigenAll, seqSpecies) = (uRecTarget(EigenAll, seqSpecies) - uRecBase(EigenAll, seqSpecies)) * beta + uRecBase(EigenAll, seqSpecies);
                    }
                }
            };
            // Only compress flow and raw rhoY polynomial coefficients; RANS
            // retains its independent scalar policy above.
            if (uRecBeta[iCell](0) < 1)
                applyFlowSpeciesBeta(uRecBeta[iCell](0));

            // PP POLICY -- final published-point guarantee. The analytic raw
            // beta is only the construction used to choose the coefficient
            // scale. Re-sample the actual coefficient storage, then enforce the
            // clipped composition that will be passed to EOS/T. Raw rhoY stays
            // in the polynomial for advection/diffusion and is not a second
            // physical acceptance invariant. Compatibility drops a clipped
            // failure directly to the base; the optional policy backtracks.
            for (int iteration = 0; iteration < 1000; ++iteration)
            {
                const auto sampled = (quadBase * uRec[iCell]).rowwise() + u[iCell].transpose();
                const bool clippedGood = clippedReferenceGood(sampled);
                if (clippedGood)
                    break;
                const real previous = uRecBeta[iCell](0);
                if (settings.uRecClippedReferencePolicy == "fallbackToBase")
                    uRecBeta[iCell](0) = 0;
                else
                    uRecBeta[iCell](0) *= .9;
                if (uRecBeta[iCell](0) < betaCutOff)
                    uRecBeta[iCell](0) = 0;
                DNDS_assert_info(previous > 0 || clippedGood,
                                 "validated reconstruction base violates its final physical-state predicate");
                applyFlowSpeciesBeta(uRecBeta[iCell](0));
            }

            // Final physical invariant uses the exact clipped composition
            // consumed by EOS/T. The raw-affine beta remains a documented
            // limiter construction, not an extra gate on the retained rhoY
            // polynomial used by conservative species fluxes.
            recInc = quadBase * uRec[iCell];
            recVRhoG = recInc.rowwise() + u[iCell].transpose();
            DNDS_assert_info(clippedReferenceGood(recVRhoG),
                             "published reconstruction is invalid after temporary composition clipping");
            if (uRecBeta[iCell](0) < 1)
            {
                ++nLimLocal;
                minBetaLocal = std::min(uRecBeta[iCell](0), minBetaLocal);
            }
        }
        if (flag & EvaluateURecBeta_LOCAL_STATS)
        {
            nLim = nLimLocal;
            betaMin = minBetaLocal;
        }
        else
        {
            MPI::Allreduce(&nLimLocal, &nLim, 1, DNDS_MPI_INDEX, MPI_SUM, u.father->getMPI().comm);
            MPI::Allreduce(&minBetaLocal, &betaMin, 1, DNDS_MPI_REAL, MPI_MIN, u.father->getMPI().comm);
        }
    }

    template <EulerModel model>
    bool EulerEvaluator<model>::AssertMeanValuePP(
        ArrayDOFV<nVarsFixed> &u, bool panic)
    {
        auto meshRead = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto meshLegacy = meshRead.LegacyHostView();
        const auto *mesh = &meshLegacy;
        this->ValidateMeshDependencies();
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
     *  @par PP POLICY
     *  Evaluate the complete coupled trial `u+alpha*res`, but derive no
     *  independent/dependent species-simplex alpha. Mixture reference energy
     *  uses the temporary clipped/normalized composition from
     *  `mixtureBaseInternalRhoE`, matching point thermodynamics. Stored `u` and
     *  `res` are never clipped or mutated here. The resulting one scalar is
     *  applied by the caller to every flow/species/RANS component (or to the
     *  common dTau). This forecast is heuristic; final stored-mean admissibility
     *  belongs to `CompressInc`. See docs/architecture/CauchyPositivity.md.
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
        auto meshRead = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto meshLegacy = meshRead.LegacyHostView();
        const auto *mesh = &meshLegacy;
        this->ValidateMeshDependencies();
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
     *  @par PP POLICY
     *  This operation only redistributes an alpha already computed by
     *  `EvaluateCellRHSAlpha`. It adds no species-simplex, thermodynamic, or
     *  storage-mutation rule of its own; final accepted-state repair remains in
     *  `CompressInc`/`AddFixedIncrement`.
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
        auto meshRead = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto meshLegacy = meshRead.LegacyHostView();
        const auto *mesh = &meshLegacy;
        this->ValidateMeshDependencies();
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
            MPI::AllreduceOneReal(rhoMin, MPI_MIN, this->mesh->getMPI());
            MPI::AllreduceOneReal(rhoEiMin, MPI_MIN, this->mesh->getMPI());
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
        auto meshRead = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto meshLegacy = meshRead.LegacyHostView();
        const auto *mesh = &meshLegacy;
        this->ValidateMeshDependencies();
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
        auto meshRequirements = Geom::MeshReadRequirements::CellFlux();
        auto meshRead = mesh->AcquireRead(meshRequirements);
        const auto meshLegacy = meshRead.LegacyHostView();
        const auto *mesh = &meshLegacy;
        this->ValidateMeshDependencies();
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        for (Geom::t_index i = Geom::BC_ID_DEFAULT_MAX; i < pBCHandler->size(); i++) // init code, consider adding to ctor
        {
            if (pBCHandler->GetFlagFromIDSoft(i, "anchorOpt") != 2)
                continue;
            if (!profileRecorders.count(i))
            {
                real RMin = veryLargeReal;
                real RMax = -veryLargeReal;
                profileRecorders.emplace(std::make_pair(i, OneDimProfile<nVarsFixed>(this->mesh->getMPI())));
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
                MPI::AllreduceOneReal(RMin, MPI_MIN, this->mesh->getMPI());
                MPI::AllreduceOneReal(RMax, MPI_MAX, this->mesh->getMPI());
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
