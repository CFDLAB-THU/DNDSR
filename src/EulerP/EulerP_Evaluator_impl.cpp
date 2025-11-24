#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "EulerP/EulerP.hpp"
#include "EulerP/EulerP_Physics.hpp"
#include "Geom/BoundaryCondition.hpp"
#include "EulerP_Evaluator_impl.hpp"
#include "EulerP_Evaluator_impl_common.hxx"

namespace DNDS::EulerP
{
    static constexpr DeviceBackend B = DeviceBackend::Host;

    template <>
    void Evaluator_impl<B>::RecGradient_GGRec(RecGradient_Arg &arg)
    {
        using namespace Geom;

        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)

        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(u)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalar)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalarGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(faceBCBuffer)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(faceBCScalarBuffer)

        int nVarsScalar = uScalar.size();
        int nVars = nVarsFlow + nVarsScalar;

        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

        DNDS_assert(faceBCBuffer.father.data());
        DNDS_assert(faceBCBuffer.father.Size() >= mesh.NumFace());

        /*********************** */
        // bc handling
        for (index iBnd = 0; iBnd < mesh.NumBnd(); iBnd++)
        {
            RecGradient_GGRec_Kernel_BndVal(self_view, arg.portable, iBnd, nVars);
        }

        /*********************** */
        // rec
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < mesh.NumCell(); iCell++)
        {
            RecGradient_GGRec_Kernel_GG(self_view, arg.portable, iCell, nVars, nVarsScalar);
        }
    }

    template <>
    void Evaluator_impl<B>::RecGradient_BarthLimiter(RecGradient_Arg &arg)
    {
        using namespace Geom;

        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(u)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalar)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalarGrad)

        int nVarsScalar = uScalar.size();
        int nVars = nVarsFlow + nVarsScalar;

        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

        /*********************** */
        // limit

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < mesh.NumCell(); iCell++)
        {
            RecGradient_BarthLimiter_Kernel_FlowPart(self_view, arg.portable, iCell, nVars, nVarsScalar);
        }

        // for scalars

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < mesh.NumCell(); iCell++)
        {
            RecGradient_BarthLimiter_Kernel_ScalarPart(self_view, arg.portable, iCell, nVars, nVarsScalar);
        }
    }

    template <>
    void Evaluator_impl<B>::Cons2PrimMu(Cons2PrimMu_Arg &arg)
    {
        using namespace Geom;

        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(u)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalar)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalarGrad)
        //
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uGradPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalarPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalarGradPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(p)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(T)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(a)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(gamma)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(mu)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(muComp)

        int nVarsScalar = uScalar.size();
        int nVars = nVarsFlow + nVarsScalar;

        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iPt = 0; iPt < u.Size(); iPt++)
        {
            // uI(Seq01234) = u[iPt];
            // for (int iVarS = 0; iVarS < nVarsScalar; iVarS++)
            //     uI[nVarsFlow + iVarS] = uScalar[iVarS](iPt);

            // gradI(EigenAll, Seq01234) = u[iPt];
            // for (int iVarS = 0; iVarS < nVarsScalar; iVarS++)
            //     gradI(EigenAll, nVarsFlow + iVarS) = uScalarGrad[iVarS][iPt];
            auto uConsI = [&] DNDS_DEVICE_CALLABLE(int i) -> real &
            {
                if (i < nVarsFlow)
                    return u(iPt, i);
                else
                    return uScalar[i - nVarsFlow](iPt);
            };
            auto uPrimI = [&] DNDS_DEVICE_CALLABLE(int i) -> real &
            {
                if (i < nVarsFlow)
                    return uPrim(iPt, i);
                else
                    return uScalarPrim[i - nVarsFlow](iPt);
            };
            auto diffUConsI = [&] DNDS_DEVICE_CALLABLE(int d, int i) -> real &
            {
                if (i < nVarsFlow)
                    return uGrad[iPt](d, i);
                else
                    return uScalarGrad[i - nVarsFlow](iPt, d);
            };
            auto diffUPrimI = [&] DNDS_DEVICE_CALLABLE(int d, int i) -> real &
            {
                if (i < nVarsFlow)
                    return uGradPrim[iPt](d, i);
                else
                    return uScalarGradPrim[i - nVarsFlow](iPt, d);
            };
            phy.Cons2Prim(uConsI, uPrimI, nVars);
            phy.Cons2PrimDiff(uConsI, uPrimI,
                              diffUConsI, diffUPrimI, nVars);

            real TT = phy.Prim2Temperature(uPrimI, nVars);
            real pp = phy.Prim2Pressure(uPrimI, nVars, TT);
            auto [gammaG, aa] = phy.Prim2GammaAcousticSpeed(uPrimI, nVars, pp);
            real muTot = phy.getMuTot(uPrimI, diffUPrimI, nVars, pp, TT);

            T(iPt) = TT;
            p(iPt) = pp;
            a(iPt) = aa;
            gamma(iPt) = gammaG;
            mu(iPt) = muTot;
            // TODO: tend to muComp!!
        }
    }

    template <>
    void Evaluator_impl<B>::Cons2Prim(Cons2Prim_Arg &arg)
    {
        using namespace Geom;

        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)
        DNDS_EULERP_IMPL_ARG_GET_REF(u)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalar)
        //
        DNDS_EULERP_IMPL_ARG_GET_REF(uPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF(p)
        DNDS_EULERP_IMPL_ARG_GET_REF(T)
        DNDS_EULERP_IMPL_ARG_GET_REF(a)
        DNDS_EULERP_IMPL_ARG_GET_REF(gamma)

        int nVarsScalar = uScalar.size();
        int nVars = nVarsFlow + nVarsScalar;

        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iPt = 0; iPt < u.Size(); iPt++)
        {
            auto uConsI = [&] DNDS_DEVICE_CALLABLE(int i) -> real &
            {
                if (i < nVarsFlow)
                    return u(iPt, i);
                else
                    return uScalar[i - nVarsFlow](iPt);
            };
            auto uPrimI = [&] DNDS_DEVICE_CALLABLE(int i) -> real &
            {
                if (i < nVarsFlow)
                    return uPrim(iPt, i);
                else
                    return uScalarPrim[i - nVarsFlow](iPt);
            };
            phy.Cons2Prim(uConsI, uPrimI, nVars);

            real TT = phy.Prim2Temperature(uPrimI, nVars);
            real pp = phy.Prim2Pressure(uPrimI, nVars, TT);
            auto [gammaG, aa] = phy.Prim2GammaAcousticSpeed(uPrimI, nVars, pp);

            T(iPt) = TT;
            p(iPt) = pp;
            a(iPt) = aa;
            gamma(iPt) = gammaG;
        }
    }

    template <>
    void Evaluator_impl<B>::EstEigenDt_GetFaceLam(EstEigenDt_Arg &arg)
    {
        using namespace Geom;

        DNDS_EULERP_IMPL_ARG_GET_REF(self)
        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)
        DNDS_EULERP_IMPL_ARG_GET_REF(u)
        DNDS_EULERP_IMPL_ARG_GET_REF(muCell)
        DNDS_EULERP_IMPL_ARG_GET_REF(aCell)
        DNDS_EULERP_IMPL_ARG_GET_REF(faceLamEst)
        DNDS_EULERP_IMPL_ARG_GET_REF(faceLamVisEst)
        DNDS_EULERP_IMPL_ARG_GET_REF(deltaLamFace)

        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iFace = 0; iFace < mesh.NumFaceProc(); iFace++)
        {
            index iCellL = mesh.face2cell(iFace, 0);
            index iCellR = mesh.face2cell(iFace, 1);
            if ((0 > iCellL || iCellL >= mesh.NumCell()) &&
                (0 > iCellR || iCellR >= mesh.NumCell()) && iCellR != UnInitIndex)
            {
                faceLamEst(iFace, 0) = UnInitReal;
                faceLamEst(iFace, 1) = UnInitReal;
                faceLamEst(iFace, 2) = UnInitReal;
                faceLamVisEst(iFace) = UnInitReal;
                deltaLamFace(iFace) = UnInitReal;
                continue; // skip if either side is not needed
            }
            TU uL = u[iCellL];
            TU uR = uL;
            tPoint nFace = fv.GetFaceNorm(iFace, -1);
            real volL = fv.GetCellVol(iCellL);
            real volR = volL;
            real vnL = uL(Seq123).dot(nFace) / uL[0];
            real vnR = uR(Seq123).dot(nFace) / uR[0];
            real aL = aCell(iCellL);
            real aR = aL;
            real muFace = muCell(iCellL);
            if (iCellR != UnInitIndex)
            {
                uR = u[iCellR];
                aR = aCell(iCellR);
                muFace = 0.5 * (muFace + muCell(iCellR));
                volR = fv.GetCellVol(iCellR);
            }
            faceLamEst(iFace, 0) = std::max(std::abs(vnL - aL), std::abs(vnR - aR));
            faceLamEst(iFace, 1) = std::max(std::abs(vnL), std::abs(vnR));
            faceLamEst(iFace, 2) = std::max(std::abs(vnL + aL), std::abs(vnR + aR));
            real rhoMean = 0.5 * (uL[0] + uR[0]);
            real nuVis = muFace / rhoMean * 2.;
            faceLamVisEst(iFace) = nuVis * fv.GetFaceArea(iFace) *
                                   (1. / volL + 1. / volR);

            //! need to verify this
            deltaLamFace(iFace) = std::max({std::abs(vnL - aL - vnR + aR),
                                            std::abs(vnL - vnR),
                                            std::abs(vnL - aL - vnR + aR)});
        }
    }

    template <>
    void Evaluator_impl<B>::EstEigenDt_FaceLam2CellDt(EstEigenDt_Arg &arg)
    {

        DNDS_EULERP_IMPL_ARG_GET_REF(self)
        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)
        DNDS_EULERP_IMPL_ARG_GET_REF(faceLamEst)
        DNDS_EULERP_IMPL_ARG_GET_REF(faceLamVisEst)
        DNDS_EULERP_IMPL_ARG_GET_REF(deltaLamFace)
        DNDS_EULERP_IMPL_ARG_GET_REF(deltaLamCell)
        DNDS_EULERP_IMPL_ARG_GET_REF(dt)

        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < mesh.NumCell(); iCell++)
        {
            real lambdaCellC = 0.0;
            real dLambdaCellC = 0.0;
            auto c2f = mesh.cell2face.father[iCell];
            for (auto iFace : c2f)
            {
                real lambdaFace =
                    std::max({faceLamEst(iFace, 0),
                              faceLamEst(iFace, 1),
                              faceLamEst(iFace, 2)}) +
                    faceLamVisEst(iFace);
                lambdaCellC += lambdaFace * fv.GetFaceArea(iFace);
                dLambdaCellC = std::max(dLambdaCellC, deltaLamFace(iFace));
            }
            dt(iCell) = fv.GetCellVol(iCell) * fv.GetCellSmoothScaleRatio(iCell) /
                        (lambdaCellC + verySmallReal);
            deltaLamCell(iCell) = dLambdaCellC;
        }
    }

    template <>
    void Evaluator_impl<B>::RecFace2nd(RecFace2nd_Arg &arg)
    {
        using namespace Geom;
        DNDS_EULERP_IMPL_ARG_GET_REF(self)
        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)

        DNDS_EULERP_IMPL_ARG_GET_REF(u)
        DNDS_EULERP_IMPL_ARG_GET_REF(uGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalar)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarGrad)

        DNDS_EULERP_IMPL_ARG_GET_REF(uFL)
        DNDS_EULERP_IMPL_ARG_GET_REF(uFR)
        DNDS_EULERP_IMPL_ARG_GET_REF(uGradFF)

        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarFL)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarFR)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarGradFF)

        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

        int nVarsScalar = uScalar.size();
        int nVars = nVarsScalar + nVarsFlow;

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iFace = 0; iFace < mesh.NumFaceProc(); iFace++)
        {
            index iCellL = mesh.face2cell(iFace, 0);
            index iCellR = mesh.face2cell(iFace, 1);
            if ((0 > iCellL || iCellL >= mesh.NumCell()) &&
                (0 > iCellR || iCellR >= mesh.NumCell()) && iCellR != UnInitIndex)
            {
                //?
                continue; // skip if either side is not needed
            }
            tPoint xrel_L = (fv.GetFaceQuadraturePPhys(iFace, -1) -
                             fv.GetCellQuadraturePPhys(iCellL, -1));
            tPoint xrel_R;
            tPoint faceNorm = fv.GetFaceNorm(iFace, -1);
            real faceLScale = veryLargeReal;
            if (iCellR != UnInitIndex)
            {
                // relative to cellR!
                xrel_R = (fv.GetFaceQuadraturePPhysFromCell(iFace, iCellR, 1, -1) -
                          fv.GetCellQuadraturePPhys(iCellR, -1));
                faceLScale = (fv.GetOtherCellBaryFromCell(iCellL, iCellR, iFace) - fv.GetCellBary(iCellL)).norm();
            }

            uFL[iFace].noalias() = u[iCellL] + uGrad[iCellL].transpose() * xrel_L;
            uGradFF[iFace] = uGrad[iCellL];
            if (iCellR != UnInitIndex)
            {
                uFR[iFace].noalias() = u[iCellR] + uGrad[iCellR].transpose() * xrel_R;
                uGradFF[iFace].noalias() += uGrad[iCellR];
                uGradFF[iFace] *= 0.5;
                uGradFF[iFace].noalias() += faceNorm * (uFR[iFace] - uFL[iFace]).transpose() * (1. / faceLScale);
            }

            for (int iVarS = 0; iVarS < nVarsScalar; iVarS++)
            {
                uScalarFL[iVarS](iFace) = uScalarGrad[iVarS][iCellL].dot(xrel_L);
                uScalarGradFF[iVarS][iFace] = uScalarGrad[iVarS][iCellL];
                if (iCellR != UnInitIndex)
                {
                    uScalarFR[iVarS](iFace) = uScalarGrad[iVarS][iCellL].dot(xrel_L);
                    uScalarGradFF[iVarS][iFace].noalias() += uScalarGrad[iVarS][iCellR];
                    uScalarGradFF[iVarS][iFace] *= 0.5;
                    uScalarGradFF[iVarS][iFace].noalias() += faceNorm * (uScalarFR[iVarS](iFace) - uScalarFL[iVarS](iFace)) * (1. / faceLScale);
                }
            }

            auto uL = [&] DNDS_DEVICE_CALLABLE(int i) -> real &
            {
                if (i < nVarsFlow)
                    return uFL(iFace, i);
                else
                    return uScalarFL[i - nVarsFlow](iFace);
            };
            auto uR = [&] DNDS_DEVICE_CALLABLE(int i) -> real &
            {
                if (i < nVarsFlow)
                    return uFR(iFace, i);
                else
                    return uScalarFR[i - nVarsFlow](iFace);
            };
            if (iCellR == UnInitIndex)
            {
                auto bc = bcHandler.id2bc(mesh.GetFaceZone(iFace));
                bc.apply(uL, uR, nVars,
                         fv.GetFaceQuadraturePPhys(iFace, -1),
                         fv.GetFaceNorm(iFace, -1),
                         phy);
            }
            // TODO: periodic handling of vectors from cell to face
        }
    }

    template <>
    void Evaluator_impl<B>::Flux2nd(Flux2nd_Arg &arg)
    {
        using namespace Geom;
        DNDS_EULERP_IMPL_ARG_GET_REF(self)
        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)

        DNDS_EULERP_IMPL_ARG_GET_REF(u)
        DNDS_EULERP_IMPL_ARG_GET_REF(uGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalar)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarGrad)

        DNDS_EULERP_IMPL_ARG_GET_REF(uPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF(uGradPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarGradPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF(p)
        DNDS_EULERP_IMPL_ARG_GET_REF(T)
        DNDS_EULERP_IMPL_ARG_GET_REF(a)
        DNDS_EULERP_IMPL_ARG_GET_REF(gamma)
        DNDS_EULERP_IMPL_ARG_GET_REF(mu)
        DNDS_EULERP_IMPL_ARG_GET_REF(muComp)

        DNDS_EULERP_IMPL_ARG_GET_REF(uFL)
        DNDS_EULERP_IMPL_ARG_GET_REF(uFR)
        DNDS_EULERP_IMPL_ARG_GET_REF(uGradFF)

        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarFL)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarFR)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarGradFF)

        DNDS_EULERP_IMPL_ARG_GET_REF(pFL)
        DNDS_EULERP_IMPL_ARG_GET_REF(pFR)
        DNDS_EULERP_IMPL_ARG_GET_REF(deltaLamFaceFF)

        DNDS_EULERP_IMPL_ARG_GET_REF(fluxFF)
        DNDS_EULERP_IMPL_ARG_GET_REF(fluxScalarFF)
        DNDS_EULERP_IMPL_ARG_GET_REF(rhs)
        DNDS_EULERP_IMPL_ARG_GET_REF(rhsScalar)

        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

        int nVarsScalar = uScalarFL.size();
        int nVars = nVarsScalar + nVarsFlow;

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iFace = 0; iFace < mesh.NumFaceProc(); iFace++)
        {
            index iCellL = mesh.face2cell(iFace, 0);
            index iCellR = mesh.face2cell(iFace, 1);
            if ((0 > iCellL || iCellL >= mesh.NumCell()) &&
                (0 > iCellR || iCellR >= mesh.NumCell()) && iCellR != UnInitIndex)
            {
                //?
                // continue; // skip if either side is not needed
            }
            tPoint n = fv.GetFaceNorm(iFace, -1);

            auto uLm = [&] DNDS_DEVICE_CALLABLE(int i) -> real &
            {
                if (i < nVarsFlow)
                    return u(iCellL, i);
                else
                    return uScalar[i - nVarsFlow](iCellL);
            };
            auto uRm = [&] DNDS_DEVICE_CALLABLE(int i) -> real &
            {
                if (i < nVarsFlow)
                    return u(iCellR, i);
                else
                    return uScalar[i - nVarsFlow](iCellR);
            };
            auto uLmPrim = [&] DNDS_DEVICE_CALLABLE(int i) -> real &
            {
                if (i < nVarsFlow)
                    return uPrim(iCellL, i);
                else
                    return uScalarPrim[i - nVarsFlow](iCellL);
            };
            index iCellRVis = iCellR == UnInitIndex ? iCellL : iCellR;
            auto uRmPrim = [&] DNDS_DEVICE_CALLABLE(int i) -> real &
            {
                if (i < nVarsFlow)
                    return uPrim(iCellRVis, i);
                else
                    return uScalarPrim[i - nVarsFlow](iCellRVis);
            };
            tPoint veloRoe;
            real vsqrRoe{0}, HRoe{0}, rhoRoe{0}, aSqrRoe{0};

            RoeAverageNS(uLm, uRm, uLmPrim, uRmPrim, nVars, p(iCellL), p(iCellRVis),
                         phy, veloRoe, vsqrRoe, HRoe, rhoRoe, aSqrRoe);
            real veloRoeN = veloRoe.dot(n);
            real aRoe = std::sqrt(std::abs(aSqrRoe));
            real lam0 = std::abs(veloRoeN - aRoe);
            real lam123 = std::abs(veloRoeN);
            real lam4 = std::abs(veloRoeN + aRoe);

            TU UL = uFL[iFace];
            TU UR = uFR[iFace];
            TU FFlow;
            FFlow.setZero();
            RoeFluxFlow(UL, UR,
                        pFL(iFace), pFR(iFace), veloRoe,
                        vsqrRoe, 0.0 /* vgn = 0 for now*/,
                        n, aSqrRoe, aRoe,
                        HRoe, phy, lam0, lam123, lam4, FFlow);

            fluxFF[iFace] = FFlow;
            // TODO: scalar's flux
            // TODO: viscous flux
        }

        // #if defined(DNDS_DIST_MT_USE_OMP)
        // #    pragma omp parallel for schedule(runtime)
        // #endif
        for (index iCell = 0; iCell < mesh.NumCell(); iCell++)
        {
            auto c2f = mesh.cell2face.father[iCell];
            for (auto iFace : c2f)
            {
                int face_sign = mesh.CellIsFaceBack(iCell, iFace) ? 1 : -1;
                rhs[iCell] -= fluxFF[iFace] * fv.GetFaceArea(iFace) / fv.GetCellVol(iCell) * face_sign;
            }
        }
    }
}