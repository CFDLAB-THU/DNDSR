#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "EulerP/EulerP.hpp"
#include "EulerP_Evaluator_impl.hpp"

namespace DNDS::EulerP
{
    template <DeviceBackend B = DeviceBackend::Host>
    DNDS_DEVICE_CALLABLE inline void RecGradient_GGRec_Kernel_BndVal(
        EvaluatorDeviceView<B> &self_view,
        typename Evaluator_impl<B>::RecGradient_Arg::Portable &arg,
        index iBnd,
        int nVars)
    {
        using namespace Geom;
        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;
        DNDS_EULERP_IMPL_ARG_GET_REF(u)
        DNDS_EULERP_IMPL_ARG_GET_REF(uGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalar)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF(faceBCBuffer)
        DNDS_EULERP_IMPL_ARG_GET_REF(faceBCScalarBuffer)

        index iFace = mesh.bnd2face(iBnd, 0);
        index iCell = mesh.bnd2cell(iBnd, 0);
        if (Geom::FaceIDIsInternal(mesh.GetFaceZone(iFace)))
            return;
        auto uI = [&] DNDS_DEVICE_CALLABLE(int i) -> real &
        {
            if (i < nVarsFlow)
                return u(iCell, i);
            else
                return uScalar[i - nVarsFlow](iCell);
        };
        auto uOut = [&] DNDS_DEVICE_CALLABLE(int i) -> real &
        {
            if (i < nVarsFlow)
                return faceBCBuffer(iFace, i);
            else
                return faceBCScalarBuffer[i - nVarsFlow](iFace);
        };

        auto bc = bcHandler.id2bc(mesh.GetFaceZone(iFace));
        bc.apply(uI, uOut, nVars,
                 fv.GetFaceQuadraturePPhys(iFace, -1),
                 fv.GetFaceNorm(iFace, -1),
                 phy);
    }

    template <DeviceBackend B = DeviceBackend::Host>
    DNDS_DEVICE_CALLABLE inline void RecGradient_GGRec_Kernel_GG(
        EvaluatorDeviceView<B> &self_view,
        typename Evaluator_impl<B>::RecGradient_Arg::Portable &arg,
        index iCell,
        int nVars, int nVarsScalar)
    {
        using namespace Geom;
        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;
        DNDS_EULERP_IMPL_ARG_GET_REF(u)
        DNDS_EULERP_IMPL_ARG_GET_REF(uGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalar)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF(faceBCBuffer)
        DNDS_EULERP_IMPL_ARG_GET_REF(faceBCScalarBuffer)

        auto grad_flow = uGrad.father[iCell];
        for (int iVarS = 0; iVarS < nVarsScalar; iVarS++)
            uScalarGrad[iVarS].father[iCell].setZero();
        grad_flow.setZero();
        auto c2f = mesh.cell2face[iCell];
        TU uI = u[iCell];

        for (long iFace : c2f)
        {
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

    template <DeviceBackend B = DeviceBackend::Host>
    DNDS_DEVICE_CALLABLE inline void RecGradient_BarthLimiter_Kernel_FlowPart(
        EvaluatorDeviceView<B> &self_view,
        typename Evaluator_impl<B>::RecGradient_Arg::Portable &arg,
        index iCell,
        int nVars, int nVarsScalar)
    {
        using namespace Geom;
        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;
        DNDS_EULERP_IMPL_ARG_GET_REF(u)
        DNDS_EULERP_IMPL_ARG_GET_REF(uGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalar)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF(faceBCBuffer)
        DNDS_EULERP_IMPL_ARG_GET_REF(faceBCScalarBuffer)

        auto c2f = mesh.cell2face[iCell];
        TU uI = u[iCell];
        TDiffU grad = uGrad.father[iCell];
        TU uIIncMax;
        TU uIIncMin;

        TU uOtherMax;
        TU uOtherMin;

        uIIncMax.setConstant(-veryLargeReal);
        uIIncMin.setConstant(veryLargeReal);
        uOtherMax.setConstant(-veryLargeReal);
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

    template <DeviceBackend B = DeviceBackend::Host>
    DNDS_DEVICE_CALLABLE inline void RecGradient_BarthLimiter_Kernel_ScalarPart(
        EvaluatorDeviceView<B> &self_view,
        typename Evaluator_impl<B>::RecGradient_Arg::Portable &arg,
        index iCell,
        int nVars, int nVarsScalar)
    {
        using namespace Geom;
        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;
        DNDS_EULERP_IMPL_ARG_GET_REF(u)
        DNDS_EULERP_IMPL_ARG_GET_REF(uGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalar)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF(faceBCBuffer)
        DNDS_EULERP_IMPL_ARG_GET_REF(faceBCScalarBuffer)

        auto c2f = mesh.cell2face[iCell];
        TU uI;
        TU uIIncMax;
        TU uIIncMin;

        TU uOtherMax;
        TU uOtherMin;

        constexpr int bufSize = nVarsFlow; // todo: use another bufsize
        for (int iVar = 0; iVar < nVarsScalar; iVar += bufSize)
        {
            int nCur = std::min(nVarsScalar - iVar, bufSize);
            for (int iVarS = 0; iVarS < nCur; iVarS++)
                uI(iVarS) = u[iVar + iVarS](iCell);
            uIIncMax.setConstant(-veryLargeReal);
            uIIncMin.setConstant(veryLargeReal);
            uOtherMax.setConstant(-veryLargeReal);
            uOtherMin.setConstant(veryLargeReal);
            for (auto iFace : c2f)
            {
                index iCellOther = mesh.CellFaceOther(iCell, iFace);
                tPoint p = (fv.GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, -1) -
                            fv.GetCellQuadraturePPhys(iCell, -1));
                for (int iVarS = 0; iVarS < nCur; iVarS++)
                {
                    real uIncPoint = uScalarGrad[iVar + iVarS].father[iCell].dot(p);
                    uIIncMax[iVarS] = std::max(uIIncMax[iVarS], uIncPoint);
                    uIIncMin[iVarS] = std::max(uIIncMin[iVarS], uIncPoint);
                }
                if (iCellOther != UnInitIndex)
                {
                    for (int iVarS = 0; iVarS < nCur; iVarS++)
                    {
                        real uIncPoint = uScalar[iVar + iVarS](iCellOther);
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
            for (int iVarS = 0; iVarS < nCur; iVarS++)
                uScalarGrad[iVar + iVarS].father[iCell] *= std::min(uOtherMax[iVarS], uOtherMin[iVarS]);
        }
    }
}