#pragma once
#include "DNDS/Defines.hpp"
#include "FiniteVolume.hpp"
#include "DNDS/JsonUtil.hpp"
#include "Geom/Geometric.hpp"

namespace DNDS::CFV
{
    template <DeviceBackend B, bool iVarOne = false>
    DNDS_DEVICE_CALLABLE void finiteVolumeCellOpTest(
        FiniteVolume::t_deviceView<B> &fv,
        tUDof<DynamicSize>::t_deviceView<B> &u,
        tUGrad<DynamicSize, 3>::t_deviceView<B> &u_grad,
        index iCell, int iVar = UnInitRowsize)
    {
        using namespace Geom;
        auto grad = u_grad[iCell];
        tPoint gradResultOne;
        if constexpr (iVarOne)
            gradResultOne.setZero();
        else
            grad.setZero();
        auto &mesh = fv.mesh;
        // if (iCell != 0)
        //     return;
        for (rowsize ic2f = 0; ic2f < mesh.cell2face.RowSize(iCell); ic2f++)
        {
            index iFace = mesh.cell2face(iCell, ic2f);
            index iCellOther = mesh.CellFaceOther(iCell, iFace);
            rowsize if2c = mesh.CellIsFaceBack(iCell, iFace) ? 0 : 1;
            if (iCellOther == UnInitIndex)
                continue; //! ignoreing BC here
            // face-out-area-norm
            tPoint norm = fv.GetFaceNormFromCell(iFace, iCell, if2c, -1) *
                          ((if2c ? -1.0 : 1.0) * fv.GetFaceArea(iFace));
            auto uOther = u[iCellOther];
            auto uI = u[iCell];
#ifdef DNDS_USE_CUDA
            if constexpr (B == DeviceBackend::CUDA)
            {
                // printf("if2c %d\n", if2c);
                // printf("iCellOther %lld\n", iCellOther);
                if constexpr (iVarOne)
                    for (int i = 0; i < 3; i++)
                        gradResultOne(i) += norm(i) * (uOther(iVar) - uI(iVar));
                else
                    for (int iV = 0; iV < grad.cols(); iV++)
                    {
                        // printf("GetFaceNormFromCell %g\n", fv.GetFaceNormFromCell(iFace, iCell, if2c, -1).norm());
                        // printf("u[iCellOther](iE) %g\n", u[iCellOther](iE));
                        // printf("u[iCell](iE) %g\n", u[iCell](iE));
                        for (int i = 0; i < 3; i++)
                            grad(i, iV) += norm(i) * (uOther(iV) - uI(iV));
                        // //! grad({0, 1, 2}, iE) does not work on CUDA!!
                        // grad.col(iE) += norm * (uOther[iE] - uI[iE]);
                    }
            }
            else
#endif
            {
                if constexpr (iVarOne)
                    gradResultOne += norm * (uOther(iVar) - uI(iVar));
                else
                    // for (int iE = 0; iE < grad.cols(); iE++)
                    //     do_one_var(iE);
                    grad.noalias() += norm * (uOther - uI).transpose();
            }
        }
        if constexpr (iVarOne)
        {
            gradResultOne *= 1.0 / fv.GetCellVol(iCell);
            // grad.col(iVar) = gradResultOne;
            for (int i = 0; i < 3; i++)
                grad(i, iVar) = gradResultOne(i);
        }
        else
            grad *= 1.0 / fv.GetCellVol(iCell);
    }

    template <DeviceBackend B>
    void finiteVolumeCellOpTest_run(FiniteVolume::t_deviceView<B> &fv,
                                    tUDof<DynamicSize>::t_deviceView<B> &u,
                                    tUGrad<DynamicSize, 3>::t_deviceView<B> &u_grad,
                                    const t_jsonconfig &settings);
    template <DeviceBackend B>
    void finiteVolumeCellOpTest_main(
        FiniteVolume &fv,
        tUDof<DynamicSize> &u,
        tUGrad<DynamicSize, 3> &u_grad,
        const t_jsonconfig &settings)
    {
        auto u_view = u.deviceView<B>();
        auto u_grad_view = u_grad.deviceView<B>();
        auto fv_view = fv.deviceView<B>();

        finiteVolumeCellOpTest_run<B>(fv_view, u_view, u_grad_view, settings);
    }
}