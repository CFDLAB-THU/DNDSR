#pragma once
#include "FiniteVolume.hpp"

namespace DNDS::CFV
{
    template <DeviceBackend B>
    DNDS_DEVICE_CALLABLE void finiteVolumeCellOpTest(
        FiniteVolume::t_deviceView<B> &fv,
        tUDof<DynamicSize>::t_deviceView<B> &u,
        tUGrad<DynamicSize, 3>::t_deviceView<B> &u_grad,
        index iCell)
    {
        using namespace Geom;
        auto grad = u_grad[iCell];
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
            if constexpr (B == DeviceBackend::CUDA)
            {
                // printf("if2c %d\n", if2c);
                // printf("iCellOther %lld\n", iCellOther);
                tPoint norm = fv.GetFaceNormFromCell(iFace, iCell, if2c, -1) *
                              ((if2c ? -1.0 : 1.0) * fv.GetFaceArea(iFace));
                auto uOther = u[iCellOther];
                auto uI = u[iCell];
                for (int iE = 0; iE < grad.cols(); iE++)
                {
                    // printf("GetFaceNormFromCell %g\n", fv.GetFaceNormFromCell(iFace, iCell, if2c, -1).norm());
                    // printf("u[iCellOther](iE) %g\n", u[iCellOther](iE));
                    // printf("u[iCell](iE) %g\n", u[iCell](iE));
                    // for (int i = 0; i < 3; i++)
                    //     grad(i, iE) += norm(i) * (u[iCellOther](iE) - u[iCell](iE));
                    //! grad({0, 1, 2}, iE) does not work on CUDA!!
                    grad.col(iE) += norm * (uOther[iE] - uI[iE]);
                }
            }
            else
            {
                // grad.noalias() += fv.GetFaceNormFromCell(iFace, iCell, if2c, -1) * (u[iCellOther] - u[iCell]).transpose() *
                //   ((if2c ? -1.0 : 1.0) * fv.GetFaceArea(iFace));
                for (int iE = 0; iE < grad.cols(); iE++)
                    grad({0, 1, 2}, iE) += fv.GetFaceNormFromCell(iFace, iCell, if2c, -1) * (u[iCellOther](iE) - u[iCell](iE)) *
                                           ((if2c ? -1.0 : 1.0) * fv.GetFaceArea(iFace));
            }
        }
        grad *= 1.0 / fv.GetCellVol(iCell);
    }
    template <DeviceBackend B>
    void finiteVolumeCellOpTest_run(FiniteVolume::t_deviceView<B> &fv,
                                    tUDof<DynamicSize>::t_deviceView<B> &u,
                                    tUGrad<DynamicSize, 3>::t_deviceView<B> &u_grad);
    template <DeviceBackend B>
    void finiteVolumeCellOpTest_main(
        FiniteVolume &fv,
        tUDof<DynamicSize> &u,
        tUGrad<DynamicSize, 3> &u_grad)
    {
        auto u_view = u.deviceView<B>();
        auto u_grad_view = u_grad.deviceView<B>();
        auto fv_view = fv.deviceView<B>();

        finiteVolumeCellOpTest_run<B>(fv_view, u_view, u_grad_view);
    }
}