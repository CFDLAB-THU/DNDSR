#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "DNDS/Errors.hpp"
#include "EulerP/EulerP.hpp"
#include "EulerP_ARS.hpp"
#include "EulerP_Evaluator_impl.hpp"
#include "Geom/Geometric.hpp"
#include "EulerP_Evaluator_impl_utils.hpp"
namespace DNDS::EulerP
{
    template <DeviceBackend B = DeviceBackend::Host>
    DNDS_DEVICE void RecGradient_GGRec_Kernel_BndVal(
        EvaluatorDeviceView<B> &self_view,
        typename Evaluator_impl<B>::RecGradient_Arg::Portable &arg,
        index iBnd, index iBndEnd,
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

        if (iBnd >= iBndEnd)
            return;

        auto bcid = mesh.GetBndZone(iBnd);
        if (Geom::FaceIDIsInternal(bcid))
            return;

        index iFace = mesh.bnd2face(iBnd, 0);
        index iCell = mesh.bnd2cell(iBnd, 0);

        auto uI = [u, uScalar, iCell] DNDS_DEVICE(int i) mutable -> real &
        {
            if (i < nVarsFlow)
                return u(iCell, i);
            else
                return uScalar[i - nVarsFlow](iCell);
        };
        auto uOut = [faceBCBuffer, faceBCScalarBuffer, iFace] DNDS_DEVICE(int i) mutable -> real &
        {
            if (i < nVarsFlow)
                return faceBCBuffer(iFace, i);
            else
                return faceBCScalarBuffer[i - nVarsFlow](iFace);
        };

        auto bc = bcHandler.id2bc(bcid);
        bc.apply(uI, uOut, nVars,
                 fv.GetFaceQuadraturePPhys(iFace, -1),
                 fv.GetFaceNorm(iFace, -1),
                 phy);
    }
    // only for intellisense hint
    template DNDS_DEVICE void RecGradient_GGRec_Kernel_BndVal<DeviceBackend::Host>(
        EvaluatorDeviceView<DeviceBackend::Host> &self_view,
        typename Evaluator_impl<DeviceBackend::Host>::RecGradient_Arg::Portable &arg,
        index iBnd, index iBndEnd,
        int nVars, int nVarsScalar);

    template <DeviceBackend B = DeviceBackend::Host>
    DNDS_DEVICE void RecGradient_GGRec_Kernel_GG(
        EvaluatorDeviceView<B> &self_view,
        typename Evaluator_impl<B>::RecGradient_Arg::Portable &arg,
        index iCell, index iCellEnd,
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

#ifdef DNDS_USE_CUDA
        __shared__ CUDA::SharedBuffer<index, 128> buf_idx;
        __shared__ CUDA::SharedBuffer<real, 128 * (3 * nVarsFlow)> buf_val;
#endif

        TDiffU grad_flow;
        grad_flow.setZero();
        if (iCell < iCellEnd)
        {
            for (int iVarS = 0; iVarS < nVarsScalar; iVarS++)
                uScalarGrad[iVarS].father[iCell].setZero();
            auto c2f = mesh.cell2face[iCell];
            TU uI = u[iCell];

            for (long iFace : c2f)
            {
                index iCellOther = mesh.CellFaceOther(iCell, iFace);
                rowsize if2c = mesh.CellIsFaceBack(iCell, iFace) ? 0 : 1;
                tPoint norm = 0.5 * fv.GetFaceNormFromCell(iFace, iCell, if2c, -1) *
                              ((if2c ? -1.0 : 1.0) * fv.GetFaceArea(iFace) / fv.GetCellVol(iCell));
                // ! the 0.5 is because we use arithmetic mean
                // ! change to 0.5 * OtherWeight if needed

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
#ifdef DNDS_USE_CUDA
        if constexpr (B == DeviceBackend::CUDA && true)
        {
#    ifdef __CUDA_ARCH__
            // using t_BufMatrix = Eigen::Matrix<real, 3, nVarsFlow * 128>;
            // __shared__ real grad_buf_data[3 * (nVarsFlow * 128)];
            // Eigen::Map<t_BufMatrix> grad_buf(grad_buf_data);

            // __shared__ index iCellThread[128];
            // int tid = threadIdx.x;
            // int bDim = blockDim.x;
            // DNDS_HD_assert(tid < 128 && tid >= 0);
            // iCellThread[tid] = iCell;
            // grad_buf.block(0, nVarsFlow * tid, 3, nVarsFlow) = grad_flow;
            // __syncthreads();
            // for (int i = 0; i < 3 * nVarsFlow; i++)
            // {
            //     int iComp = (i * bDim + tid);
            //     int iCellInBlock = iComp / (3 * nVarsFlow);
            //     int iCompSub = iComp % (3 * nVarsFlow);
            //     // int iComp = (i * bDim + tid) % (3 * nVarsFlow);
            //     index iCellC = iCellThread[iCellInBlock];
            //     uGrad.father[iCellC](iCompSub) = grad_buf(iComp);
            // }
#    endif
            detail::CUDA_Local2GlobalAssign<3 * nVarsFlow, 128>(
                [grad_flow] DNDS_DEVICE(int i) mutable -> real &
                { return grad_flow(i); },
                [uGrad] DNDS_DEVICE(index iCellC, int i) mutable -> real &
                { return uGrad.father[iCellC](i); },
                buf_idx, buf_val,
                iCell, iCellEnd);
        }
        else
#endif
        {
            if (iCell < iCellEnd)
                uGrad.father[iCell] = grad_flow;
        }
    }
    // only for intellisense hint
    template DNDS_DEVICE void RecGradient_GGRec_Kernel_GG<DeviceBackend::Host>(
        EvaluatorDeviceView<DeviceBackend::Host> &self_view,
        typename Evaluator_impl<DeviceBackend::Host>::RecGradient_Arg::Portable &arg,
        index iCell, index iCellEnd,
        int nVars, int nVarsScalar);

    template <DeviceBackend B = DeviceBackend::Host>
    DNDS_DEVICE void RecGradient_BarthLimiter_Kernel_FlowPart(
        EvaluatorDeviceView<B> &self_view,
        typename Evaluator_impl<B>::RecGradient_Arg::Portable &arg,
        index iCell, index iCellEnd,
        int nVars, int nVarsScalar)
    {
        using namespace Geom;
        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;
        DNDS_EULERP_IMPL_ARG_GET_REF(u)
        DNDS_EULERP_IMPL_ARG_GET_REF(uGrad)

#ifdef DNDS_USE_CUDA
        __shared__ CUDA::SharedBuffer<index, 128> buf_idx;
        __shared__ CUDA::SharedBuffer<real, 128 * (3 * nVarsFlow)> buf_val;
#endif

        TDiffU grad;

        if (iCell < iCellEnd)
        {
            auto c2f = mesh.cell2face[iCell];
            TU uI = u[iCell];
            grad = uGrad.father[iCell];

            tPoint uI_mag;
            uI_mag << uI(0), uI(I4), U123(uI).norm();
            tPoint uIIncMax;
            tPoint uIIncMin;

            tPoint uOtherMax = uI_mag;
            tPoint uOtherMin = uI_mag;

            // we use 0.0 here to avoid invert situation
            uIIncMax.setConstant(0.0);
            uIIncMin.setConstant(0.0);

            real EInternalI = phy.Cons2EInternal(uI, nVarsFlow);
            real EInternalMin = EInternalI;

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                index iCellOther = mesh.CellFaceOther(iCell, iFace);
                TU uIncPoint = grad.transpose() *
                               (fv.GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, -1) -
                                fv.GetCellQuadraturePPhys(iCell, -1));
                tPoint uIncPoint_mag;
                uIncPoint_mag << uIncPoint(0), uIncPoint(I4), U123(uIncPoint).norm();
                uIIncMax = uIIncMax.array().max(uIncPoint_mag.array());
                uIIncMin = uIIncMin.array().min(uIncPoint_mag.array());

                if (iCellOther != UnInitIndex)
                {
                    uIncPoint = u[iCellOther];
                    uIncPoint_mag << uIncPoint(0), uIncPoint(I4), U123(uIncPoint).norm();
                    uOtherMax = uOtherMax.array().max(uIncPoint_mag.array());
                    uOtherMin = uOtherMin.array().min(uIncPoint_mag.array());
                    real EOther = phy.Cons2EInternal(uIncPoint, nVarsFlow);
                    EInternalMin = std::min(EOther, EInternalMin);
                }
            }

            uOtherMax -= uI_mag;
            uOtherMin -= uI_mag;
            uOtherMax = (uOtherMax.array().abs()) / (uIIncMax.array().abs() + verySmallReal);
            uOtherMin = (uOtherMin.array().abs()) / (uIIncMin.array().abs() + verySmallReal);
            uOtherMin = uOtherMin.array().min(uOtherMax.array());
            uOtherMin = uOtherMin.array().max(0.0).min(1.0);
            // grad.array().rowwise() *= (uOtherMax.array().min(uOtherMin.array())).transpose();
            // grad *= uOtherMin(0.0);
            grad.col(0) *= uOtherMin(0);
            grad.col(I4) *= uOtherMin(1);
            for (int i = 1; i < 1 + 3; i++)
                grad.col(i) *= uOtherMin(2);

            // use the new grad for EInternal reconstruction!
            real EInternalPointMin = EInternalI;
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                index iCellOther = mesh.CellFaceOther(iCell, iFace);
                TU uIncPoint = grad.transpose() *
                               (fv.GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, -1) -
                                fv.GetCellQuadraturePPhys(iCell, -1));
                uIncPoint += uI;
                real EOther = phy.Cons2EInternal(uIncPoint, nVarsFlow);
                EInternalPointMin = std::min(EOther, EInternalPointMin);
            }
            real alphaE =
                (EInternalI - EInternalMin * 0.1) / std::max(EInternalI - EInternalPointMin, verySmallReal);
            grad *= std::clamp(alphaE, 0., 1.);

            // for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            // {
            //     index iFace = c2f[ic2f];
            //     index iCellOther = mesh.CellFaceOther(iCell, iFace);
            //     TU uIncPoint = grad.transpose() *
            //                    (fv.GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, -1) -
            //                     fv.GetCellQuadraturePPhys(iCell, -1));
            //     uIncPoint += uI;
            //     real EOther = phy.Cons2EInternal(uIncPoint, nVarsFlow);
            //     DNDS_HD_assert(EOther > 0);
            // }
        }
#ifdef DNDS_USE_CUDA
        if constexpr (B == DeviceBackend::CUDA)
        {
            detail::CUDA_Local2GlobalAssign<3 * nVarsFlow, 128>(
                [grad] DNDS_DEVICE(int i) mutable -> real &
                { return grad(i); },
                [uGrad] DNDS_DEVICE(index iCellC, int i) mutable -> real &
                { return uGrad.father[iCellC](i); },
                buf_idx, buf_val,
                iCell, iCellEnd);
        }
        else
#endif
        {
            if (iCell < iCellEnd)
                uGrad.father[iCell] = grad;
        }
    }
    // only for intellisense hint
    template DNDS_DEVICE void RecGradient_BarthLimiter_Kernel_FlowPart<DeviceBackend::Host>(
        EvaluatorDeviceView<DeviceBackend::Host> &self_view,
        typename Evaluator_impl<DeviceBackend::Host>::RecGradient_Arg::Portable &arg,
        index iCell, index iCellEnd,
        int nVars, int nVarsScalar);

    template <DeviceBackend B = DeviceBackend::Host>
    DNDS_DEVICE void RecGradient_BarthLimiter_Kernel_ScalarPart(
        EvaluatorDeviceView<B> &self_view,
        typename Evaluator_impl<B>::RecGradient_Arg::Portable &arg,
        index iCell, index iCellEnd,
        int nVars, int nVarsScalar)
    {
        using namespace Geom;
        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalar)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF(faceBCBuffer)
        DNDS_EULERP_IMPL_ARG_GET_REF(faceBCScalarBuffer)

        if (iCell >= iCellEnd)
            return;

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
                uI(iVarS) = uScalar[iVar + iVarS](iCell);
            // we use 0.0 here to avoid invert situation
            uIIncMax.setConstant(-0.0);
            uIIncMin.setConstant(0.0);
            uOtherMax = uOtherMin = uI;

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
            uOtherMax = uOtherMax.array().abs() / (uIIncMax.array().abs() + verySmallReal);
            uOtherMin = uOtherMin.array().abs() / (uIIncMin.array().abs() + verySmallReal);
            uOtherMax = uOtherMax.array().max(0.0).min(1.0);
            uOtherMin = uOtherMin.array().max(0.0).min(1.0);
            for (int iVarS = 0; iVarS < nCur; iVarS++)
                uScalarGrad[iVar + iVarS].father[iCell] *= std::min(uOtherMax[iVarS], uOtherMin[iVarS]);
        }
    }
    // only for intellisense hint
    template DNDS_DEVICE void RecGradient_BarthLimiter_Kernel_ScalarPart<DeviceBackend::Host>(
        EvaluatorDeviceView<DeviceBackend::Host> &self_view,
        typename Evaluator_impl<DeviceBackend::Host>::RecGradient_Arg::Portable &arg,
        index iCell, index iCellEnd,
        int nVars, int nVarsScalar);

    template <DeviceBackend B = DeviceBackend::Host>
    DNDS_DEVICE void Cons2PrimMu_Kernel(
        EvaluatorDeviceView<B> &self_view,
        typename Evaluator_impl<B>::Cons2PrimMu_Arg::Portable &arg,
        index iPt, index iPtEnd, int nVars, int nVarsScalar)
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
        //
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

#ifdef DNDS_USE_CUDA
        __shared__ CUDA::SharedBuffer<index, 128> buf_idx;
        __shared__ CUDA::SharedBuffer<real, 128 * (3 * nVarsFlow)> buf_val;
#endif

        TU uPrimC;
        Eigen::Map<TU> uPrimCM(uPrimC.data());
        TDiffU uGradPrimC;
        Eigen::Map<TDiffU> uGradPrimCM(uGradPrimC.data());
        if (iPt < iPtEnd)
        {
            // uI(Seq01234) = u[iPt];
            // for (int iVarS = 0; iVarS < nVarsScalar; iVarS++)
            //     uI[nVarsFlow + iVarS] = uScalar[iVarS](iPt);

            // gradI(EigenAll, Seq01234) = u[iPt];
            // for (int iVarS = 0; iVarS < nVarsScalar; iVarS++)
            //     gradI(EigenAll, nVarsFlow + iVarS) = uScalarGrad[iVarS][iPt];
            auto uConsI = [u, uScalar, iPt] DNDS_DEVICE(int i) mutable -> real &
            {
                if (i < nVarsFlow)
                    return u(iPt, i);
                else
                    return uScalar[i - nVarsFlow](iPt);
            };
            auto uPrimI = [uPrimCM, uScalarPrim, iPt] DNDS_DEVICE(int i) mutable -> real &
            {
                if (i < nVarsFlow)
                    return uPrimCM(i);
                else
                    return uScalarPrim[i - nVarsFlow](iPt);
            };
            auto diffUConsI = [uGrad, uScalarGrad, iPt] DNDS_DEVICE(int d, int i) mutable -> real &
            {
                if (i < nVarsFlow)
                    return uGrad[iPt](d, i);
                else
                    return uScalarGrad[i - nVarsFlow](iPt, d);
            };
            auto diffUPrimI = [uGradPrimCM, uScalarGradPrim, iPt] DNDS_DEVICE(int d, int i) mutable -> real &
            {
                if (i < nVarsFlow)
                    return uGradPrimCM(d, i);
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
#ifdef DNDS_USE_CUDA
        if constexpr (B == DeviceBackend::CUDA)
        {
            detail::CUDA_Local2GlobalAssign<3 * nVarsFlow, 128>(
                [uGradPrimCM] DNDS_DEVICE(int i) mutable -> real &
                { return uGradPrimCM(i); },
                [uGradPrim] DNDS_DEVICE(index iPtC, int i) mutable -> real &
                { return uGradPrim[iPtC](i); },
                buf_idx, buf_val,
                iPt, iPtEnd);

            detail::CUDA_Local2GlobalAssign<nVarsFlow, 128>(
                [uPrimCM] DNDS_DEVICE(int i) mutable -> real &
                { return uPrimCM(i); },
                [uPrim] DNDS_DEVICE(index iPtC, int i) mutable -> real &
                { return uPrim[iPtC](i); },
                buf_idx, buf_val,
                iPt, iPtEnd);
        }
        else
#endif
        {
            if (iPt < iPtEnd)
            {
                uPrim[iPt] = uPrimC;
                uGradPrim[iPt] = uGradPrimC;
            }
        }
    }
    // only for intellisense hint
    template DNDS_DEVICE void Cons2PrimMu_Kernel<DeviceBackend::Host>(
        EvaluatorDeviceView<DeviceBackend::Host> &self_view,
        typename Evaluator_impl<DeviceBackend::Host>::Cons2PrimMu_Arg::Portable &arg,
        index iPt, index iPtEnd,
        int nVars, int nVarsScalar);

    template <DeviceBackend B = DeviceBackend::Host>
    DNDS_DEVICE void Cons2Prim_Kernel(
        EvaluatorDeviceView<B> &self_view,
        typename Evaluator_impl<B>::Cons2Prim_Arg::Portable &arg,
        index iPt, index iPtEnd,
        int nVars, int nVarsScalar)
    {
        using namespace Geom;
        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

        DNDS_EULERP_IMPL_ARG_GET_REF(u)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalar)
        //
        DNDS_EULERP_IMPL_ARG_GET_REF(uPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF(p)
        DNDS_EULERP_IMPL_ARG_GET_REF(T)
        DNDS_EULERP_IMPL_ARG_GET_REF(a)
        DNDS_EULERP_IMPL_ARG_GET_REF(gamma)

#ifdef DNDS_USE_CUDA
        __shared__ CUDA::SharedBuffer<index, 128> buf_idx;
        __shared__ CUDA::SharedBuffer<real, 128 * (1 * nVarsFlow)> buf_val;
#endif
        TU uPrimC;
        Eigen::Map<TU> uPrimCM(uPrimC.data());
        if (iPt < iPtEnd)
        {
            auto uConsI = [u, uScalar, iPt] DNDS_DEVICE(int i) mutable -> real &
            {
                if (i < nVarsFlow)
                    return u(iPt, i);
                else
                    return uScalar[i - nVarsFlow](iPt);
            };
            auto uPrimI = [uPrimCM, uScalarPrim, iPt] DNDS_DEVICE(int i) mutable -> real &
            {
                if (i < nVarsFlow)
                    return uPrimCM(i);
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
#ifdef DNDS_USE_CUDA
        if constexpr (B == DeviceBackend::CUDA)
        {
            detail::CUDA_Local2GlobalAssign<nVarsFlow, 128>(
                [uPrimCM] DNDS_DEVICE(int i) mutable -> real &
                { return uPrimCM(i); },
                [uPrim] DNDS_DEVICE(index iPtC, int i) mutable -> real &
                { return uPrim[iPtC](i); },
                buf_idx, buf_val,
                iPt, iPtEnd);
        }
        else
#endif
        {
            if (iPt < iPtEnd)
            {
                uPrim[iPt] = uPrimC;
            }
        }
    }
    // only for intellisense hint
    template DNDS_DEVICE void Cons2Prim_Kernel<DeviceBackend::Host>(
        EvaluatorDeviceView<DeviceBackend::Host> &self_view,
        typename Evaluator_impl<DeviceBackend::Host>::Cons2Prim_Arg::Portable &arg,
        index iPt, index iPtEnd,
        int nVars, int nVarsScalar);

    template <DeviceBackend B = DeviceBackend::Host>
    DNDS_DEVICE void EstEigenDt_GetFaceLam_Kernel(
        EvaluatorDeviceView<B> &self_view,
        typename Evaluator_impl<B>::EstEigenDt_Arg::Portable &arg,
        index iFace, index iFaceEnd,
        int nVars, int nVarsScalar)
    {
        using namespace Geom;
        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

        DNDS_EULERP_IMPL_ARG_GET_REF(u)
        DNDS_EULERP_IMPL_ARG_GET_REF(muCell)
        DNDS_EULERP_IMPL_ARG_GET_REF(aCell)
        DNDS_EULERP_IMPL_ARG_GET_REF(faceLamEst)
        DNDS_EULERP_IMPL_ARG_GET_REF(faceLamVisEst)
        DNDS_EULERP_IMPL_ARG_GET_REF(deltaLamFace)

        if (iFace >= iFaceEnd)
            return;

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
            // return; // skip if either side is not needed
        }
        TU uL = u[iCellL];
        TU uR = uL;
        tPoint nFace = fv.GetFaceNorm(iFace, -1);
        real volL = fv.GetCellVol(iCellL);
        real volR = volL;
        real vnL = U123(uL).dot(nFace) / uL[0];
        real vnR = U123(uR).dot(nFace) / uR[0];
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

    // only for intellisense hint
    template DNDS_DEVICE void EstEigenDt_GetFaceLam_Kernel<DeviceBackend::Host>(
        EvaluatorDeviceView<DeviceBackend::Host> &self_view,
        typename Evaluator_impl<DeviceBackend::Host>::EstEigenDt_Arg::Portable &arg,
        index iFace, index iFaceEnd,
        int nVars, int nVarsScalar);

    template <DeviceBackend B = DeviceBackend::Host>
    DNDS_DEVICE void EstEigenDt_FaceLam2CellDt_Kernel(
        EvaluatorDeviceView<B> &self_view,
        typename Evaluator_impl<B>::EstEigenDt_Arg::Portable &arg,
        index iCell, index iCellEnd,
        int nVars, int nVarsScalar)
    {
        using namespace Geom;
        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

        DNDS_EULERP_IMPL_ARG_GET_REF(faceLamEst)
        DNDS_EULERP_IMPL_ARG_GET_REF(faceLamVisEst)
        DNDS_EULERP_IMPL_ARG_GET_REF(deltaLamFace)
        DNDS_EULERP_IMPL_ARG_GET_REF(deltaLamCell)
        DNDS_EULERP_IMPL_ARG_GET_REF(dt)

        if (iCell >= iCellEnd)
            return;

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

    // only for intellisense hint
    template DNDS_DEVICE void EstEigenDt_FaceLam2CellDt_Kernel<DeviceBackend::Host>(
        EvaluatorDeviceView<DeviceBackend::Host> &self_view,
        typename Evaluator_impl<DeviceBackend::Host>::EstEigenDt_Arg::Portable &arg,
        index iCell, index iCellEnd,
        int nVars, int nVarsScalar);

    template <DeviceBackend B = DeviceBackend::Host>
    DNDS_DEVICE void RecFace2nd_Kernel(
        EvaluatorDeviceView<B> &self_view,
        typename Evaluator_impl<B>::RecFace2nd_Arg::Portable &arg,
        index iFace, index iFaceEnd,
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

        DNDS_EULERP_IMPL_ARG_GET_REF(uFL)
        DNDS_EULERP_IMPL_ARG_GET_REF(uFR)
        DNDS_EULERP_IMPL_ARG_GET_REF(uGradFF)

        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarFL)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarFR)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarGradFF)

#ifdef DNDS_USE_CUDA
        __shared__ CUDA::SharedBuffer<index, 128> buf_idx;
        __shared__ CUDA::SharedBuffer<real, 128 * (3 * nVarsFlow)> buf_val;
#endif

        TDiffU uGradFFC;
        Eigen::Map<TDiffU> uGradFFCM(uGradFFC.data());

        if (iFace < iFaceEnd)
        {
            index iCellL = mesh.face2cell(iFace, 0);
            index iCellR = mesh.face2cell(iFace, 1);
            if ((0 > iCellL || iCellL >= mesh.NumCell()) &&
                (0 > iCellR || iCellR >= mesh.NumCell()) && iCellR != UnInitIndex)
            {
                //?
                // return; // skip if either side is not needed
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
            uGradFFC = uGrad[iCellL];
            if (iCellR != UnInitIndex)
            {
                uFR[iFace].noalias() = u[iCellR] + uGrad[iCellR].transpose() * xrel_R;
                uGradFFC.noalias() += uGrad[iCellR];
                uGradFFC *= 0.5;
                uGradFFC.noalias() += faceNorm * (uFR[iFace] - uFL[iFace]).transpose() * (1. / faceLScale);
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

            auto uL = [uFL, uScalarFL, iFace] DNDS_DEVICE(int i) mutable -> real &
            {
                if (i < nVarsFlow)
                    return uFL(iFace, i);
                else
                    return uScalarFL[i - nVarsFlow](iFace);
            };
            auto uR = [uFR, uScalarFR, iFace] DNDS_DEVICE(int i) mutable -> real &
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
#ifdef DNDS_USE_CUDA
        if constexpr (B == DeviceBackend::CUDA)
        {
            detail::CUDA_Local2GlobalAssign<3 * nVarsFlow, 128>(
                [uGradFFCM] DNDS_DEVICE(int i) mutable -> real &
                { return uGradFFCM(i); },
                [uGradFF] DNDS_DEVICE(index iFaceC, int i) mutable -> real &
                { return uGradFF[iFaceC](i); },
                buf_idx, buf_val,
                iFace, iFaceEnd);
        }
        else
#endif
        {
            if (iFace < iFaceEnd)
            {
                uGradFF[iFace] = uGradFFC;
            }
        }
    }

    // only for intellisense hint
    template DNDS_DEVICE void RecFace2nd_Kernel<DeviceBackend::Host>(
        EvaluatorDeviceView<DeviceBackend::Host> &self_view,
        typename Evaluator_impl<DeviceBackend::Host>::RecFace2nd_Arg::Portable &arg,
        index iFace, index iFaceEnd,
        int nVars, int nVarsScalar);

    template <DeviceBackend B = DeviceBackend::Host>
    DNDS_DEVICE void Flux2nd_Kernel_FluxFace(
        EvaluatorDeviceView<B> &self_view,
        typename Evaluator_impl<B>::Flux2nd_Arg::Portable &arg,
        index iFace, index iFaceEnd,
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
        DNDS_EULERP_IMPL_ARG_GET_REF(deltaLamCell)

        DNDS_EULERP_IMPL_ARG_GET_REF(uFL)
        DNDS_EULERP_IMPL_ARG_GET_REF(uFR)
        DNDS_EULERP_IMPL_ARG_GET_REF(uGradFF)

        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarFL)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarFR)
        DNDS_EULERP_IMPL_ARG_GET_REF(uScalarGradFF)

        DNDS_EULERP_IMPL_ARG_GET_REF(pFL)
        DNDS_EULERP_IMPL_ARG_GET_REF(pFR)

        DNDS_EULERP_IMPL_ARG_GET_REF(fluxFF)
        DNDS_EULERP_IMPL_ARG_GET_REF(fluxScalarFF)
        // DNDS_EULERP_IMPL_ARG_GET_REF(rhs)
        // DNDS_EULERP_IMPL_ARG_GET_REF(rhsScalar)

#ifdef DNDS_USE_CUDA
        __shared__ CUDA::SharedBuffer<index, 128> buf_idx;
        __shared__ CUDA::SharedBuffer<real, 128 * (1 * nVarsFlow)> buf_val;
#endif

        TU FFlow;
        Eigen::Map<TU> FFlowM(FFlow.data());

        if (iFace < iFaceEnd)
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

            index iCellRR = iCellR == UnInitIndex ? iCellL : iCellR;
            auto uLm = [u, uScalar, iCellL] DNDS_DEVICE(int i) mutable -> real &
            {
                if (i < nVarsFlow)
                    return u(iCellL, i);
                else
                    return uScalar[i - nVarsFlow](iCellL);
            };
            auto uRm = [u, uScalar, iCellRR] DNDS_DEVICE(int i) mutable -> real &
            {
                if (i < nVarsFlow)
                    return u(iCellRR, i);
                else
                    return uScalar[i - nVarsFlow](iCellRR);
            };
            auto uLmPrim = [uPrim, uScalarPrim, iCellL] DNDS_DEVICE(int i) mutable -> real &
            {
                if (i < nVarsFlow)
                    return uPrim(iCellL, i);
                else
                    return uScalarPrim[i - nVarsFlow](iCellL);
            };
            auto uRmPrim = [uPrim, uScalarPrim, iCellRR] DNDS_DEVICE(int i) mutable -> real &
            {
                if (i < nVarsFlow)
                    return uPrim(iCellRR, i);
                else
                    return uScalarPrim[i - nVarsFlow](iCellRR);
            };
            tPoint veloRoe;
            real vsqrRoe{0}, HRoe{0}, rhoRoe{0}, aSqrRoe{0};

            RoeAverageNS(uLm, uRm, uLmPrim, uRmPrim, nVars, p(iCellL), p(iCellRR),
                         phy, veloRoe, vsqrRoe, HRoe, rhoRoe, aSqrRoe);
            real veloRoeN = veloRoe.dot(n);
            real aRoe = std::sqrt(std::abs(aSqrRoe));
            real lam0 = std::abs(veloRoeN - aRoe);
            real lam123 = std::abs(veloRoeN);
            real lam4 = std::abs(veloRoeN + aRoe);

            tPoint vL = U123(uPrim[iCellL]);
            tPoint vR = U123(uPrim[iCellRR]);
            real aL = a(iCellL);
            real aR = a(iCellRR);
            real pL = p(iCellL);
            real pR = p(iCellRR);
            TU UL = uFL[iFace];
            TU UR = uFR[iFace];

            RoeEigenValueFixer(
                aRoe, aRoe, veloRoeN, veloRoeN,
                std::max(deltaLamCell(iCellL), deltaLamCell(iCellRR)),
                1.0, lam0, lam123, lam4);

            real lamm = std::max(aL + std::abs(vL.dot(n)), aR + std::abs(vR.dot(n)));
            // lam0 = lam123 = lam4 = lamm;

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
#ifdef DNDS_USE_CUDA
        if constexpr (B == DeviceBackend::CUDA)
        {
            detail::CUDA_Local2GlobalAssign<1 * nVarsFlow, 128>(
                [FFlowM] DNDS_DEVICE(int i) mutable -> real &
                { return FFlowM(i); },
                [fluxFF] DNDS_DEVICE(index iFaceC, int i) mutable -> real &
                { return fluxFF[iFaceC](i); },
                buf_idx, buf_val,
                iFace, iFaceEnd);
        }
        else
#endif
        {
            if (iFace < iFaceEnd)
            {
                fluxFF[iFace] = FFlow;
            }
        }
    }

    // only for intellisense hint
    template DNDS_DEVICE void Flux2nd_Kernel_FluxFace<DeviceBackend::Host>(
        EvaluatorDeviceView<DeviceBackend::Host> &self_view,
        typename Evaluator_impl<DeviceBackend::Host>::Flux2nd_Arg::Portable &arg,
        index iFace, index iFaceEnd,
        int nVars, int nVarsScalar);

    template <DeviceBackend B = DeviceBackend::Host>
    DNDS_DEVICE void Flux2nd_Kernel_Face2Cell(
        EvaluatorDeviceView<B> &self_view,
        typename Evaluator_impl<B>::Flux2nd_Arg::Portable &arg,
        index iCell, index iCellEnd,
        int nVars, int nVarsScalar)
    {
        using namespace Geom;
        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

        DNDS_EULERP_IMPL_ARG_GET_REF(fluxFF)
        DNDS_EULERP_IMPL_ARG_GET_REF(fluxScalarFF)
        DNDS_EULERP_IMPL_ARG_GET_REF(rhs)
        DNDS_EULERP_IMPL_ARG_GET_REF(rhsScalar)

#ifdef DNDS_USE_CUDA
        __shared__ CUDA::SharedBuffer<index, 128> buf_idx;
        __shared__ CUDA::SharedBuffer<real, 128 * (1 * nVarsFlow)> buf_val;
#endif

        TU rhsC;
        Eigen::Map<TU> rhsCM(rhsC.data());

        if (iCell < iCellEnd)
        {
            rhsC = rhs[iCell];
            auto c2f = mesh.cell2face.father[iCell];
            for (auto iFace : c2f)
            {
                real face_sign = mesh.CellIsFaceBack(iCell, iFace) ? 1 : -1;
                real face_coef = -fv.GetFaceArea(iFace) / fv.GetCellVol(iCell) * face_sign;
                rhsC += fluxFF[iFace] * face_coef;
                for (int iVarS = 0; iVarS < nVarsScalar; iVarS++)
                    rhsScalar[iVarS](iCell) += fluxScalarFF[iVarS](iFace) * face_coef;
            }
        }
#ifdef DNDS_USE_CUDA
        if constexpr (B == DeviceBackend::CUDA)
        {
            detail::CUDA_Local2GlobalAssign<1 * nVarsFlow, 128>(
                [rhsCM] DNDS_DEVICE(int i) mutable -> real &
                { return rhsCM(i); },
                [rhs] DNDS_DEVICE(index iCellC, int i) mutable -> real &
                { return rhs[iCellC](i); },
                buf_idx, buf_val,
                iCell, iCellEnd);
        }
        else
#endif
        {
            if (iCell < iCellEnd)
            {
                rhs[iCell] = rhsC;
            }
        }
    }

    // only for intellisense hint
    template DNDS_DEVICE void Flux2nd_Kernel_Face2Cell<DeviceBackend::Host>(
        EvaluatorDeviceView<DeviceBackend::Host> &self_view,
        typename Evaluator_impl<DeviceBackend::Host>::Flux2nd_Arg::Portable &arg,
        index iCell, index iCellEnd,
        int nVars, int nVarsScalar);
}