#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "EulerP_Evaluator_impl.hpp"

namespace DNDS::EulerP
{
    static constexpr DeviceBackend B = DeviceBackend::Host;

    template <>
    void Evaluator_impl<B>::RecGradient_GGRec(
        RecGradient_Arg &arg)
    {
        using namespace Geom;

        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)
        DNDS_EULERP_IMPL_ARG_GET_REF(u)
        DNDS_EULERP_IMPL_ARG_GET_REF(uGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalar)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF(faceBCBuffer)
        DNDS_EULERP_IMPL_ARG_GET_REF(faceBCScalarBuffer)

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
        VectorXR fullUOther;
        VectorXR fullU;
        fullU.resize(nVars);
        fullUOther.resize(nVars);
        for (index iBnd = 0; iBnd < mesh.NumBnd(); iBnd++)
        {
            index iFace = mesh.bnd2face(iBnd, 0);
            index iCell = mesh.bnd2cell(iBnd, 0);
            fullU(Seq01234) = u[iCell];
            for (int iVarS = 0; iVarS < nVarsScalar; iVarS++)
                fullU(iVarS + nVarsFlow) = uScalar[iVarS](iCell);
            auto bc = bcHandler.id2bc(mesh.GetFaceZone(iFace));
            bc.apply(fullU.data(), fullUOther.data(), nVars,
                     fv.GetFaceQuadraturePPhys(iFace, -1).data(),
                     fv.GetFaceNorm(iFace, -1).data(),
                     phy);
            faceBCBuffer[iCell] = fullUOther(Seq01234);
            for (int iVarS = 0; iVarS < nVarsScalar; iVarS++)
                faceBCScalarBuffer[iVarS](iFace) = fullUOther(iVarS + nVarsFlow);
        }

        /*********************** */
        // rec
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < mesh.NumCell(); iCell++)
        {
            auto grad_flow = uGrad.father[iCell];
            for (int iVarS = 0; iVarS < nVarsScalar; iVarS++)
                uScalarGrad[iVarS].father[iCell].setZero();
            grad_flow.setZero();
            auto c2f = mesh.cell2face[iCell];
            TU uI = u[iCell];

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                index iCellOther = mesh.CellFaceOther(iCell, iFace);
                rowsize if2c = mesh.CellIsFaceBack(iCell, iFace) ? 0 : 1;
                tPoint norm = fv.GetFaceNormFromCell(iFace, iCell, if2c, -1) *
                              ((if2c ? -1.0 : 1.0) * fv.GetFaceArea(iFace));

                TU uOther;
                uOther.setZero();
                if (iCellOther != UnInitIndex)
                    uOther = u[iCellOther];
                else
                    uOther = faceBCBuffer.father[iFace];

                grad_flow.noalias() += norm * (uOther - uI).transpose();
                for (int iVarS = 0; iVarS < nVarsScalar; iVarS++)
                {
                    real uI = uScalar[iVarS].father(iCell);
                    real uOther = 0.;
                    if (iCellOther != UnInitIndex)
                        uOther = uScalar[iVarS](iCell);
                    else
                        uOther = faceBCScalarBuffer[iVarS](iCell);
                    uScalarGrad[iVarS].father[iCell].noalias() += norm * (uOther - uI);
                }
            }
        }
    }

    template <>
    void Evaluator_impl<B>::RecGradient_BarthLimiter(RecGradient_Arg &arg)
    {
        using namespace Geom;

        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)
        DNDS_EULERP_IMPL_ARG_GET_REF(u)
        DNDS_EULERP_IMPL_ARG_GET_REF(uGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalar)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarGrad)

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
            auto c2f = mesh.cell2face[iCell];
            TU uI = u[iCell];
            TDiffU grad = uGrad.father[iCell];
            TU uIIncMax;
            TU uIIncMin;

            TU uOtherMax;
            TU uOtherMin;

            uIIncMax.setConstant(verySmallReal);
            uIIncMin.setConstant(veryLargeReal);
            uOtherMax.setConstant(verySmallReal);
            uOtherMin.setConstant(veryLargeReal);

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                index iCellOther = mesh.CellFaceOther(iCell, iFace);
                TU uIncPoint = grad.transpose() *
                               (fv.GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, -1) -
                                fv.GetCellQuadraturePPhys(iCell, -1));
                uIIncMax = uIIncMax.array().max(uIncPoint.array());
                uIIncMin = uIIncMin.array().min(uIncPoint.array());
                if (iCellOther != UnInitIndex)
                {
                    uIncPoint = u[iCellOther];
                    uOtherMax = uOtherMax.array().max(uIncPoint.array());
                    uOtherMin = uOtherMin.array().min(uIncPoint.array());
                }
            }

            uOtherMax -= uI;
            uOtherMin -= uI;
            uOtherMax = uOtherMax.array() / (uIIncMax.array().abs() + verySmallReal);
            uOtherMin = -uOtherMin.array() / (uIIncMin.array().abs() + verySmallReal);
            uOtherMax = uOtherMax.array().max(0.0).min(1.0);
            uOtherMin = uOtherMin.array().max(0.0).min(1.0);
            grad.array().rowwise() *= (uOtherMax.array().min(uOtherMin.array())).transpose();
        }

        // for scalars

        VectorXR uIIncMax;
        uIIncMax.resize(nVarsScalar);
        VectorXR uIIncMin;
        uIIncMin.resize(nVarsScalar);

        VectorXR uOtherMax;
        uOtherMax.resize(nVarsScalar);
        VectorXR uOtherMin;
        uOtherMin.resize(nVarsScalar);

        VectorXR uI;
        uI.resize(nVarsScalar);

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime) firstprivate(uIIncMax, uIIncMin, uOtherMax, uOtherMin, uI)
#endif
        for (index iCell = 0; iCell < mesh.NumCell(); iCell++)
        {
            auto c2f = mesh.cell2face[iCell];
            for (int iVar = 0; iVar < nVarsScalar; iVar++)
                uI[iVar] = uScalar[iVar].father(iCell);
            uIIncMax.setConstant(verySmallReal);
            uIIncMin.setConstant(veryLargeReal);
            uOtherMax.setConstant(verySmallReal);
            uOtherMin.setConstant(veryLargeReal);

            for (auto iFace : c2f)
            {
                index iCellOther = mesh.CellFaceOther(iCell, iFace);
                tPoint p = (fv.GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, -1) -
                            fv.GetCellQuadraturePPhys(iCell, -1));
                for (int iVar = 0; iVar < nVarsScalar; iVar++)
                {
                    real uIncPoint = uScalarGrad[iVar].father[iCell].dot(p);
                    uIIncMax[iVar] = std::max(uIIncMax[iVar], uIncPoint);
                    uIIncMin[iVar] = std::max(uIIncMin[iVar], uIncPoint);
                }
                if (iCellOther != UnInitIndex)
                {
                    for (int iVarS = 0; iVarS < nVarsScalar; iVarS++)
                    {
                        real uIncPoint = uScalar[iVarS](iCellOther);
                        uOtherMax[iVarS] = std::max(uIIncMax[iVarS], uIncPoint);
                        uOtherMin[iVarS] = std::min(uIIncMin[iVarS], uIncPoint);
                    }
                }
            }

            uOtherMax -= uI;
            uOtherMin -= uI;
            uOtherMax = uOtherMax.array() / (uIIncMax.array().abs() + verySmallReal);
            uOtherMin = -uOtherMin.array() / (uIIncMin.array().abs() + verySmallReal);
            uOtherMax = uOtherMax.array().max(0.0).min(1.0);
            uOtherMin = uOtherMin.array().max(0.0).min(1.0);
            for (int iVar = 0; iVar < nVarsScalar; iVar++)
                uScalarGrad[iVar].father[iCell] *= std::min(uOtherMax[iVar], uOtherMin[iVar]);
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
}