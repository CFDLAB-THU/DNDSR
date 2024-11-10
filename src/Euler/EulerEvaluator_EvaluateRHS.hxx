#pragma once
#include "EulerEvaluator.hpp"
#include <fmt/core.h>

namespace DNDS::Euler
{
    /**
     * @details
     * about RHS:
     * with topology fixed, RHS is dependent on:
     * flux:
     *      dofs:  u_L u_R, urec_L, urec_R,
     *      geoms: dbv_l, dbv_r, detJacobian_f, uNorm_f,
     */
#define IF_NOT_NOREC (1)
    DNDS_SWITCH_INTELLISENSE(
        // the real definition
        template <EulerModel model>
        ,
        // the intellisense friendly definition
        static const auto model = NS_SA;
        template <>
    )
    void EulerEvaluator<model>::EvaluateRHS(
        ArrayDOFV<nVarsFixed> &rhs,
        JacobianDiagBlock<nVarsFixed> &JSource,
        ArrayDOFV<nVarsFixed> &u,
        ArrayRECV<nVarsFixed> &uRecUnlim,
        ArrayRECV<nVarsFixed> &uRec,
        ArrayDOFV<1> &uRecBeta,
        ArrayDOFV<1> &cellRHSAlpha,
        bool onlyOnHalfAlpha,
        real t,
        uint64_t flags)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "EvaluateRHS 1");
        int cnvars = nVars;
        auto rsType = settings.rsType;
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            rhs[iCell].setZero();
        }
        TU fluxWallSumLocal;
        fluxWallSumLocal.setZero(cnvars);
        fluxWallSum.setZero(cnvars);
        nFaceReducedOrder = 0;

        bool ignoreVis = flags & RHS_Ignore_Viscosity;

        for (Geom::t_index i = Geom::BC_ID_DEFAULT_MAX; i < pBCHandler->size(); i++) // init code, consider adding to ctor
        {
            if (pBCHandler->GetFlagFromIDSoft(i, "integrationOpt") == 0)
                continue;
            if (!bndIntegrations.count(i))
            {
                auto intOpt = pBCHandler->GetFlagFromIDSoft(i, "integrationOpt");
                bndIntegrations.emplace(std::make_pair(i, IntegrationRecorder(mesh->getMPI(), intOpt == 1 ? nVars : nVars + 2)));
            }
        }
        for (auto &v : bndIntegrations)
            v.second.Reset();

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

        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            auto f2c = mesh->face2cell[iFace];
            auto gFace = vfv->GetFaceQuad(iFace);
#ifdef USE_FLUX_BALANCE_TERM
            Eigen::Matrix<real, nVarsFixed, 3, Eigen::ColMajor> fluxEs(cnvars, 3);
#else
            Eigen::Matrix<real, nVarsFixed, 1, Eigen::ColMajor> fluxEs(cnvars, 1);
#endif
            if (onlyOnHalfAlpha)
            {
                // bool lIsHalfAlpha = cellIsHalfAlpha(f2c[0]);
                // bool rIsHalfAlpha =
                //     (f2c[1] != UnInitIndex && f2c[1] < mesh->NumCell()) // must be owned cell!
                //         ? cellIsHalfAlpha(f2c[0])
                //         : false;
                // if (!(lIsHalfAlpha || rIsHalfAlpha))
                //     continue;
            }

            fluxEs.setZero();

            // auto f2n = mesh->face2node[iFace];
            // Geom::tSmallCoords coords;
            // mesh->LoadCoords(f2n, coords);

            Geom::Elem::SummationNoOp noOp;
            auto faceBndID = mesh->GetFaceZone(iFace);
            auto faceBCType = pBCHandler->GetTypeFromID(faceBndID);
            TU_Batch ULxyV, URxyV;
            ULxyV.resize(u[f2c[0]].size(), gFace.GetNumPoints());
            URxyV.resizeLike(ULxyV);
            TDiffU_Batch DiffUxyV, DiffUxyPrimV;
            DiffUxyV.resize(dim * gFace.GetNumPoints(), u[f2c[0]].size());
            DiffUxyPrimV.resizeLike(DiffUxyV);
            TVec_Batch unitNormV, vgXYV;
            unitNormV.resize(dim, gFace.GetNumPoints()), vgXYV.resizeLike(unitNormV);

            TVec unitNormCent = vfv->GetFaceNorm(iFace, -1)(Seq012);
            TMat normBaseCent = Geom::NormBuildLocalBaseV<dim>(unitNormCent);
            TU ULMeanXy = u[f2c[0]];
            this->UFromCell2Face(ULMeanXy, iFace, f2c[0], 0);
            TU URMeanXy;
            if (f2c[1] != UnInitIndex)
            {
                URMeanXy = u[f2c[1]];
                this->UFromCell2Face(URMeanXy, iFace, f2c[1], 1);
            }
            else
            {
                URMeanXy = generateBoundaryValue(
                    ULMeanXy, ULMeanXy, f2c[0], iFace, -1,
                    unitNormCent,
                    normBaseCent,
                    vfv->GetFaceQuadraturePPhys(iFace, -1),
                    t,
                    mesh->GetFaceZone(iFace), false, 0);
            }
            TU_Batch FLFix, FRFix;
#ifdef USE_FLUX_BALANCE_TERM
            FLFix.setZero(cnvars, gFace.GetNumPoints()), FRFix.setZero(cnvars, gFace.GetNumPoints()); // todo: finish in faceFlux
#endif

            gFace.IntegrationSimple(
                noOp,
                [&](decltype(noOp) &finc, int iG)
                {
                    // finc.resizeLike(fluxEs);
                    int nDiff = vfv->GetFaceAtr(iFace).NDIFF;
                    TVec unitNorm = vfv->GetFaceNorm(iFace, iG)(Seq012);
                    TMat normBase = Geom::NormBuildLocalBaseV<dim>(unitNorm);
                    PerformanceTimer::Instance().StartTimer(PerformanceTimer::LimiterB);

                    TU ULxy = u[f2c[0]];
                    ULxy += (vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, iG, std::array<int, 1>{0}, 1) *
                             uRec[f2c[0]])
                                .transpose() *
                            IF_NOT_NOREC;
                    this->UFromCell2Face(ULxy, iFace, f2c[0], 0);
                    TU ULxyUnlim;
                    if (&uRecUnlim != &uRec)
                    {
                        ULxyUnlim = u[f2c[0]];
                        ULxyUnlim += (vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, iG, std::array<int, 1>{0}, 1) *
                                      uRecUnlim[f2c[0]])
                                         .transpose() *
                                     IF_NOT_NOREC;
                        this->UFromCell2Face(ULxyUnlim, iFace, f2c[0], 0);
                    }
                    else
                        ULxyUnlim = ULxy;

                    TU URxy, URxyUnlim;
#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
                    TDiffU GradULxy, GradURxy;
                    GradULxy.resize(Eigen::NoChange, cnvars);
                    GradURxy.resize(Eigen::NoChange, cnvars);
                    GradULxy.setZero(), GradURxy.setZero();

                    if constexpr (gDim == 2)
                        GradULxy({0, 1}, Eigen::all) =
                            vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, iG, std::array<int, 2>{1, 2}, 3) *
                            uRecUnlim[f2c[0]] * IF_NOT_NOREC; // 2d here
                    else
                        GradULxy({0, 1, 2}, Eigen::all) =
                            vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, iG, std::array<int, 3>{1, 2, 3}, 4) *
                            uRecUnlim[f2c[0]] * IF_NOT_NOREC; // 3d here
                    this->DiffUFromCell2Face(GradULxy, iFace, f2c[0], 0);
                    if (ignoreVis)
                        GradULxy *= 0.;

#endif
                    real minVol = vfv->GetCellVol(f2c[0]);
                    // DNDS_MPI_InsertCheck(u.father->getMPI(), "RHS inner 2");

                    if (f2c[1] != UnInitIndex)
                    {
                        URxy = u[f2c[1]];
                        URxy += (vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, iG, std::array<int, 1>{0}, 1) *
                                 uRec[f2c[1]])
                                    .transpose() *
                                IF_NOT_NOREC;
                        this->UFromCell2Face(URxy, iFace, f2c[1], 1);
                        if (&uRecUnlim != &uRec)
                        {
                            URxyUnlim = u[f2c[1]];
                            URxyUnlim += (vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, iG, std::array<int, 1>{0}, 1) *
                                          uRecUnlim[f2c[1]])
                                             .transpose() *
                                         IF_NOT_NOREC;
                            this->UFromCell2Face(URxyUnlim, iFace, f2c[1], 1);
                        }
                        else
                            URxyUnlim = URxy;

#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
                        if constexpr (gDim == 2)
                            GradURxy({0, 1}, Eigen::all) =
                                vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, iG, std::array<int, 2>{1, 2}, 3) *
                                uRecUnlim[f2c[1]] * IF_NOT_NOREC; // 2d here
                        else
                            GradURxy({0, 1, 2}, Eigen::all) =
                                vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, iG, std::array<int, 3>{1, 2, 3}, 4) *
                                uRecUnlim[f2c[1]] * IF_NOT_NOREC; // 3d here
                        this->DiffUFromCell2Face(GradURxy, iFace, f2c[1], 1);
                        if (ignoreVis)
                            GradURxy *= 0.;
#endif
                        minVol = std::min(minVol, vfv->GetCellVol(f2c[1]));
                    }
                    else if (true) // is bc
                    {
                        URxy = generateBoundaryValue(
                            ULxy, ULMeanXy, f2c[0], iFace, iG,
                            unitNorm,
                            normBase,
                            vfv->GetFaceQuadraturePPhys(iFace, iG),
                            t,
                            mesh->GetFaceZone(iFace), true, 0);
                        if (&uRecUnlim != &uRec && false) //! disabled now, as ULxyUnlim may have negative pressure failing in generateBV
                            URxyUnlim = generateBoundaryValue(
                                ULxyUnlim, ULMeanXy, f2c[0], iFace, iG,
                                unitNorm,
                                normBase,
                                vfv->GetFaceQuadraturePPhys(iFace, iG),
                                t,
                                mesh->GetFaceZone(iFace), true, 0);
                        else
                            URxyUnlim = URxy;
#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
                        GradURxy = GradULxy;
#endif
                    }
                    PerformanceTimer::Instance().StopTimer(PerformanceTimer::LimiterB);

                    real distGRP = minVol / vfv->GetFaceArea(iFace) * 2;
#ifdef USE_DISABLE_DIST_GRP_FIX_AT_WALL
                    distGRP += faceBCType == EulerBCType::BCWall ? veryLargeReal : 0.0;
#endif
                    distGRP += faceBCType == EulerBCType::BCWallInvis ? veryLargeReal : 0.0;
                    distGRP += faceBCType == EulerBCType::BCSym ? veryLargeReal : 0.0;
                    TU UMeanXy = 0.5 * (ULxy + URxy);

#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
                    TDiffU GradUMeanXy = (GradURxy + GradULxy) * 0.5 +
                                         (1.0 / distGRP) *
                                             (unitNorm * (URxyUnlim - ULxyUnlim).transpose());
                    if (ignoreVis)
                        GradUMeanXy *= 0.;

                    TDiffU GradUMeanXyPrim;
                    real gamma = settings.idealGasProperty.gamma;
                    if (settings.usePrimGradInVisFlux)
                    {
                        TDiffU GradULxyPrim, GradURxyPrim;
                        GradULxyPrim.resizeLike(GradURxy), GradURxyPrim.resizeLike(GradURxy);
                        Gas::GradientCons2Prim_IdealGas<dim>(ULxy, GradULxy, GradULxyPrim, gamma);
                        Gas::GradientCons2Prim_IdealGas<dim>(URxy, GradURxy, GradURxyPrim, gamma);
                        TU URxyPrim(cnvars), ULxyPrim(cnvars);
                        Gas::IdealGasThermalConservative2Primitive(ULxy, ULxyPrim, gamma);
                        Gas::IdealGasThermalConservative2Primitive(URxy, URxyPrim, gamma);

                        GradUMeanXyPrim = (GradURxyPrim + GradULxyPrim) * 0.5 +
                                          (1.0 / distGRP) *
                                              (unitNorm * (URxyPrim - ULxyPrim).transpose());
                    }
                    else
                        Gas::GradientCons2Prim_IdealGas(UMeanXy, GradUMeanXy, GradUMeanXyPrim, gamma);

#else
                    TDiffU GradUMeanXy;
#endif
                    if (faceBCType == EulerBCType::BCWallInvis ||
                        // faceBCType == EulerBCType::BCIn ||
                        faceBCType == EulerBCType::BCOut ||
                        faceBCType == EulerBCType::BCFar ||
                        faceBCType == EulerBCType::BCSpecial ||
                        faceBCType == EulerBCType::BCSym)
                        GradUMeanXy *= 0, GradUMeanXyPrim *= 0; // force no viscid flux

                    if (!GradUMeanXy.allFinite())
                    {
                        std::cout << GradURxy << std::endl;
                        std::cout << GradULxy << std::endl;
                        std::cout << distGRP << std::endl;
                        std::cout << f2c[0] << " " << f2c[1] << " " << mesh->NumCell() << " " << mesh->NumCellProc() << std::endl;
                        std::cout << uRec[f2c[0]] << std::endl;
                        std::cout << "-----------------------------------\n";
                        if (f2c[1] != UnInitIndex)
                            std::cout << uRec[f2c[1]] << std::endl;
                        std::cout << "-----------------------------------\n";
                        std::cout << u[f2c[0]].transpose() << std::endl;
                        if (f2c[1] != UnInitIndex)
                            std::cout << u[f2c[1]].transpose() << std::endl;
                        DNDS_assert(false);
                    }

                    auto seqC = Eigen::seq(iG * dim, iG * dim + dim - 1);
                    ULxyV(Eigen::all, iG) = ULxy;
                    URxyV(Eigen::all, iG) = URxy;
                    DiffUxyV(seqC, Eigen::all) = GradUMeanXy;
                    DiffUxyPrimV(seqC, Eigen::all) = GradUMeanXyPrim;
                    unitNormV(Eigen::all, iG) = unitNorm;
                    vgXYV(Eigen::all, iG) = GetFaceVGrid(iFace, iG);
                });
            TU_Batch fincC = fluxFace(
                ULxyV, URxyV,
                ULMeanXy, URMeanXy,
                DiffUxyV, DiffUxyPrimV,
                unitNormV,
                vgXYV,
                unitNormCent,
                GetFaceVGrid(iFace, -1),
                FLFix,
                FRFix,
                mesh->GetFaceZone(iFace),
                rsType,
                iFace);

            gFace.IntegrationSimple(
                fluxEs,
                [&](decltype(fluxEs) &finc, int iG)
                {
                    finc.resizeLike(fluxEs);
                    finc(Eigen::all, 0) = fincC(Eigen::all, iG);
#ifdef USE_FLUX_BALANCE_TERM
                    finc(Eigen::all, 1) = FLFix(Eigen::all, iG);
                    finc(Eigen::all, 2) = FRFix(Eigen::all, iG);
#endif
                    finc *= vfv->GetFaceJacobiDet(iFace, iG); // !don't forget this
                });

            TU fluxIncL = fluxEs(Eigen::all, 0);
            TU fluxIncR = -fluxEs(Eigen::all, 0);
#ifdef USE_FLUX_BALANCE_TERM
            fluxIncL -= fluxEs(Eigen::all, 1);
            fluxIncR += fluxEs(Eigen::all, 2);
#endif
            this->UFromFace2Cell(fluxIncL, iFace, f2c[0], 0);
            if (f2c[1] != UnInitIndex)
                this->UFromFace2Cell(fluxIncR, iFace, f2c[1], 1); // periodic back to cell
            real alphaFace = cellRHSAlpha[f2c[0]](0);
            if (f2c[1] != UnInitIndex)
                alphaFace = std::min(alphaFace, cellRHSAlpha[f2c[1]](0));

            rhs[f2c[0]] += fluxIncL / vfv->GetCellVol(f2c[0]);
            if (f2c[1] != UnInitIndex)
                rhs[f2c[1]] += fluxIncR / vfv->GetCellVol(f2c[1]);

            if (faceBCType == EulerBCType::BCWall || // TODO: update to general
                (faceBCType == EulerBCType::BCWallInvis && settings.idealGasProperty.muGas < 1e-99))
            {
                fluxWallSumLocal -= fluxEs(Eigen::all, 0);
                if (iFace >= mesh->NumFace())
                    DNDS_assert(false);
            }
            if (f2c[1] == UnInitIndex)
            {
                DNDS_assert(mesh->face2bnd.find(iFace) != mesh->face2bnd.end());
                fluxBnd.at(mesh->face2bnd[iFace]) = fluxEs(Eigen::all, 0) / vfv->GetFaceArea(iFace);
                TVec fluxBndForceTInt;
                fluxBndForceTInt.setZero();
                gFace.IntegrationSimple(
                    fluxBndForceTInt,
                    [&](decltype(fluxBndForceTInt) &finc, int iG)
                    {
                        TU fcur = fincC(Eigen::all, iG);
                        TVec ncur = unitNormV(Eigen::all, iG);
                        finc = fcur(Seq123);
                        finc -= ncur * (ncur.dot(finc));
                        finc *= vfv->GetFaceJacobiDet(iFace, iG); // !don't forget this
                    });
                fluxBndForceT.at(mesh->face2bnd[iFace]) = fluxBndForceTInt / vfv->GetFaceArea(iFace);
            }

            // integrations
            if (pBCHandler->GetFlagFromIDSoft(mesh->GetFaceZone(iFace), "integrationOpt") == 1)
            {
                bndIntegrations.at(mesh->GetFaceZone(iFace)).Add(fluxEs(Eigen::all, 0), vfv->GetFaceArea(iFace));
            }
            if (pBCHandler->GetFlagFromIDSoft(mesh->GetFaceZone(iFace), "integrationOpt") == 2)
            {
                TU uL = u[f2c[0]];
                if (settings.frameConstRotation.enabled)
                    this->TransformURotatingFrame(uL, vfv->GetFaceQuadraturePPhys(iFace, -1), 1);
                TU uLPrim = uL;
                auto gamma = settings.idealGasProperty.gamma;
                Gas::IdealGasThermalConservative2Primitive(uL, uLPrim, gamma);
                Eigen::Vector<real, Eigen::Dynamic> vInt;
                vInt.resize(nVars + 2);
                vInt(Eigen::seq(0, nVars - 1)) = uL;

                auto [p0, T0] = Gas::IdealGasThermalPrimitiveGetP0T0<dim>(uLPrim, gamma, settings.idealGasProperty.Rgas);
                vInt(nVars) = p0, vInt(nVars + 1) = T0;
                vInt(0) = 1;
                bndIntegrations.at(mesh->GetFaceZone(iFace)).Add(vInt * fluxEs(0, 0), fluxEs(0, 0));
            }
        }

        DNDS_MPI_InsertCheck(u.father->getMPI(), "EvaluateRHS After Flux");

        if (!settings.ignoreSourceTerm)
        {

            JSource.clearValues();

            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                if (onlyOnHalfAlpha)
                {
                    //     if (!cellIsHalfAlpha(iCell))
                    //         continue;
                }
                auto gCell = vfv->GetCellQuad(iCell);

                Eigen::Matrix<real, nVarsFixed, Eigen::Dynamic> sourceV; // now includes sourcejacobian diag
                sourceV.setZero(cnvars, JSource.isBlock() ? cnvars + 1 : 2);

                Geom::Elem::SummationNoOp noOp;
                bool cellOrderReduced = false;

                gCell.IntegrationSimple(
                    sourceV,
                    [&](decltype(sourceV) &finc, int iG)
                    {
                        TDiffU GradU;
                        GradU.resize(Eigen::NoChange, cnvars);
                        GradU.setZero();
                        PerformanceTimer::Instance().StartTimer(PerformanceTimer::LimiterB);
                        if constexpr (gDim == 2)
                            GradU({0, 1}, Eigen::all) =
                                vfv->GetIntPointDiffBaseValue(iCell, -1, -1, iG, std::array<int, 2>{1, 2}, 3) *
                                uRecUnlim[iCell] * IF_NOT_NOREC; // 2d specific
                        else
                            GradU({0, 1, 2}, Eigen::all) =
                                vfv->GetIntPointDiffBaseValue(iCell, -1, -1, iG, std::array<int, 3>{1, 2, 3}, 4) *
                                uRecUnlim[iCell] * IF_NOT_NOREC; // 3d specific

                        bool pointOrderReduced;
                        TU ULxy = u[iCell];
                        if (!cellOrderReduced)
                        {
                            // ULxy += cellDiBjGaussBatchElemVR.m(ig).row(0).rightCols(uRec[iCell].rows()) *
                            //         uRec[iCell] * IF_NOT_NOREC;
                            ULxy +=
                                (vfv->GetIntPointDiffBaseValue(iCell, -1, -1, iG, std::array<int, 1>{0}, 1) *
                                 uRec[iCell])
                                    .transpose() *
                                IF_NOT_NOREC;
                        }
                        PerformanceTimer::Instance().StopTimer(PerformanceTimer::LimiterB);

                        // bool compressed = false;
                        // ULxy = CompressRecPart(u[iCell], ULxy, compressed); //! do not forget the mean value

                        finc.resizeLike(sourceV);
                        TJacobianU jac;
                        finc(Eigen::all, 0) =
                            source(
                                ULxy,
                                GradU,
                                vfv->GetCellQuadraturePPhys(iCell, iG), jac,
                                iCell, iG, 0);
                        TU sourceJDiag =
                            source(
                                ULxy,
                                GradU,
                                vfv->GetCellQuadraturePPhys(iCell, iG), jac,
                                iCell, iG, JSource.isBlock() ? 2 : 1);
                        if (JSource.isBlock())
                            finc(Eigen::all, Eigen::seq(Eigen::fix<1>, Eigen::last)) = jac;
                        else
                            finc(Eigen::all, 1) = sourceJDiag;

                        finc *= vfv->GetCellJacobiDet(iCell, iG); // don't forget this
                        if (finc.hasNaN() || (!finc.allFinite()))
                        {
                            std::cout << finc.transpose() << std::endl;
                            std::cout << ULxy.transpose() << std::endl;
                            std::cout << GradU << std::endl;
                            DNDS_assert(false);
                        }
                    });
                sourceV *= cellRHSAlpha[iCell](0) / vfv->GetCellVol(iCell); // becomes mean value
                rhs[iCell] += sourceV(Eigen::all, 0);
                if (JSource.isBlock())
                    JSource.getBlock(iCell) = sourceV(Eigen::all, Eigen::seq(Eigen::fix<1>, Eigen::last));
                else
                    JSource.getDiag(iCell) = sourceV(Eigen::all, 1);

                // if (iCell == 18195)
                // {
                //     std::cout << rhs[iCell].transpose() << std::endl;
                // }
            }
        }
        // quick aux: reduce the wall flux sum
        MPI::Allreduce(fluxWallSumLocal.data(), fluxWallSum.data(), fluxWallSum.size(), DNDS_MPI_REAL, MPI_SUM, u.father->getMPI().comm);
        for (auto &i : bndIntegrations)
        {

            i.second.Reduce();
        }

        DNDS_MPI_InsertCheck(u.father->getMPI(), "EvaluateRHS -1");
    }
}
