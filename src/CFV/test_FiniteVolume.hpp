#pragma once
#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "DNDS/Errors.hpp"
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
        index iCell, real *local_buf, int iVar = UnInitRowsize)
    {
        using namespace Geom;
        auto grad = u_grad[iCell];
        tPoint gradResultOne;
        Eigen::Map<Eigen::Matrix<real, 3, DynamicSize>> gradResult(local_buf, 3, grad.cols());
        if constexpr (iVarOne)
            gradResultOne.setZero();
        else
            DNDS_HD_assert(local_buf),
                gradResult.setZero();
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
                            gradResult(i, iV) += norm(i) * (uOther(iV) - uI(iV));
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
                    gradResult.noalias() += norm * (uOther - uI).transpose();
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
        {
            gradResult *= 1.0 / fv.GetCellVol(iCell);
            grad = gradResult;
        }
    }

    template <DeviceBackend B>
    void finiteVolumeCellOpTest_run(FiniteVolume::t_deviceView<B> &fv,
                                    tUDof<DynamicSize>::t_deviceView<B> &u,
                                    tUGrad<DynamicSize, 3>::t_deviceView<B> &u_grad,
                                    int nIter,
                                    const t_jsonconfig &settings);
    template <DeviceBackend B>
    void finiteVolumeCellOpTest_main(
        FiniteVolume &fv,
        tUDof<DynamicSize> &u,
        tUGrad<DynamicSize, 3> &u_grad,
        int nIter,
        const t_jsonconfig &settings)
    {
        auto u_view = u.deviceView<B>();
        auto u_grad_view = u_grad.deviceView<B>();
        auto fv_view = fv.deviceView<B>();

        finiteVolumeCellOpTest_run<B>(fv_view, u_view, u_grad_view, nIter, settings);
    }
    /*************************************************************/

    /*************************************************************/

    /*************************************************************/

    /*************************************************************/

    template <DeviceBackend B, int nVarsFixed, bool iVarOne = false>
    DNDS_DEVICE_CALLABLE void finiteVolumeCellOpTest_Fixed(
        FiniteVolume::t_deviceView<B> &fv,
        typename tUDof<nVarsFixed>::template t_deviceView<B> &u,
        typename tUGrad<nVarsFixed, 3>::template t_deviceView<B> &u_grad,
        index iCell, int iVar = UnInitRowsize)
    {
        static_assert(nVarsFixed > 0);
        using namespace Geom;
        auto grad = u_grad[iCell];
        tPoint gradResultOne;
        Eigen::Matrix<real, 3, nVarsFixed> gradResult;
        if constexpr (iVarOne)
            gradResultOne.setZero();
        else
            gradResult.setZero();
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
                if constexpr (iVarOne)
                    for (int i = 0; i < 3; i++)
                        gradResultOne(i) += norm(i) * (uOther(iVar) - uI(iVar));
                else
                    for (int iV = 0; iV < grad.cols(); iV++)
                    {
                        for (int i = 0; i < 3; i++)
                            gradResult(i, iV) += norm(i) * (uOther(iV) - uI(iV));
                        // //! grad({0, 1, 2}, iE) does not work on CUDA!!
                        // grad.col(iE) += norm * (uOther[iE] - uI[iE]);
                    }
            }
            else
#endif
            {
                if constexpr (iVarOne)
                    gradResultOne.noalias() += norm * (uOther(iVar) - uI(iVar));
                else
                    gradResult.noalias() += norm * (uOther - uI).transpose();
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
        {
            gradResult *= 1.0 / fv.GetCellVol(iCell);
            grad = gradResult;
        }
    }

    template <DeviceBackend B, int nVarsFixed>
    struct finiteVolumeCellOpTest_Fixed_entry;

    template <int nVarsFixed>
    struct finiteVolumeCellOpTest_Fixed_entry<DeviceBackend::CUDA, nVarsFixed>
    {
        static constexpr auto B = DeviceBackend::CUDA;
        void run(FiniteVolume::t_deviceView<B> &fv,
                 typename tUDof<nVarsFixed>::template t_deviceView<B> &u,
                 typename tUGrad<nVarsFixed, 3>::template t_deviceView<B> &u_grad,
                 int nIter,
                 const t_jsonconfig &settings);
    };

    template <int nVarsFixed>
    struct finiteVolumeCellOpTest_Fixed_entry<DeviceBackend::Host, nVarsFixed>
    {
        static constexpr auto B = DeviceBackend::Host;
        void run(FiniteVolume::t_deviceView<B> &fv,
                 typename tUDof<nVarsFixed>::template t_deviceView<B> &u,
                 typename tUGrad<nVarsFixed, 3>::template t_deviceView<B> &u_grad,
                 int nIter,
                 const t_jsonconfig &settings);
    };

    template <DeviceBackend B, int nVarsFixed>
    void finiteVolumeCellOpTest_Fixed_main(
        FiniteVolume &fv,
        tUDof<nVarsFixed> &u,
        tUGrad<nVarsFixed, 3> &u_grad,
        int nIter,
        const t_jsonconfig &settings)
    {
        auto u_view = u.template deviceView<B>();
        auto u_grad_view = u_grad.template deviceView<B>();
        auto fv_view = fv.deviceView<B>();

        finiteVolumeCellOpTest_Fixed_entry<B, nVarsFixed>().run(fv_view, u_view, u_grad_view, nIter, settings);
    }

    /*************************************************************/

    /*************************************************************/

    /*************************************************************/

    /*************************************************************/

    template <DeviceBackend B, int nVarsFixed, bool iVarOne = false>
    DNDS_DEVICE_CALLABLE void finiteVolumeCellOpTest_SOA_ver0(
        FiniteVolume::t_deviceView<B> &fv,
        std::array<tUDof<1>::t_deviceView<B>, nVarsFixed> &u,
        std::array<tUGrad<1, 3>::t_deviceView<B>, nVarsFixed> &u_grad,
        index iCell, int iVar = UnInitRowsize)
    {
        static_assert(nVarsFixed > 0);
        using namespace Geom;
        tPoint gradResultOne;
        Eigen::Matrix<real, 3, nVarsFixed> gradResult;
        if constexpr (iVarOne)
            gradResultOne.setZero();
        else
            gradResult.setZero();
        auto &mesh = fv.mesh;
        // if (iCell != 0)
        //     return;
        for (rowsize ic2f = 0; ic2f < mesh.cell2face.RowSize(iCell); ic2f++)
        {
            index iFace = mesh.cell2face(iCell, ic2f);
            index iCellOther = mesh.CellFaceOther(iCell, iFace);
            rowsize if2c = mesh.CellIsFaceBack(iCell, iFace) ? 0 : 1;
            if (iCellOther == UnInitIndex)
                continue; //! ignoring BC here
            // face-out-area-norm
            tPoint norm = fv.GetFaceNormFromCell(iFace, iCell, if2c, -1) *
                          ((if2c ? -1.0 : 1.0) * fv.GetFaceArea(iFace));

            if constexpr (iVarOne)
                for (int i = 0; i < 3; i++)
                    gradResultOne(i) += norm(i) * (u[iVar][iCellOther](0) - u[iVar][iCell](0));
            else
                for (int iV = 0; iV < nVarsFixed; iV++)
                {
                    for (int i = 0; i < 3; i++)
                        gradResult(i, iV) += norm(i) * (u[iV][iCellOther](0) - u[iV][iCell](0));
                    // //! grad({0, 1, 2}, iE) does not work on CUDA!!
                    // grad.col(iE) += norm * (uOther[iE] - uI[iE]);
                }
        }
        if constexpr (iVarOne)
        {
            gradResultOne *= 1.0 / fv.GetCellVol(iCell);
            // grad.col(iVar) = gradResultOne;
            for (int i = 0; i < 3; i++)
                u_grad[iVar][iCell](i, 0) = gradResultOne(i);
        }
        else
        {
            gradResult *= 1.0 / fv.GetCellVol(iCell);
            for (int iV = 0; iV < nVarsFixed; iV++)
                for (int i = 0; i < 3; i++)
                    u_grad[iV][iCell](i) = gradResult(i, iV);
        }
    }

    template <DeviceBackend B, int nVarsFixed>
    struct finiteVolumeCellOpTest_SOA_ver0_entry;

    template <int nVarsFixed>
    struct finiteVolumeCellOpTest_SOA_ver0_entry<DeviceBackend::CUDA, nVarsFixed>
    {
        static constexpr auto B = DeviceBackend::CUDA;
        void run(FiniteVolume::t_deviceView<B> &fv,
                 std::array<tUDof<1>::t_deviceView<B>, nVarsFixed> &u,
                 std::array<tUGrad<1, 3>::t_deviceView<B>, nVarsFixed> &u_grad,
                 int nIter,
                 const t_jsonconfig &settings);
    };

    template <int nVarsFixed>
    struct finiteVolumeCellOpTest_SOA_ver0_entry<DeviceBackend::Host, nVarsFixed>
    {
        static constexpr auto B = DeviceBackend::Host;
        void run(FiniteVolume::t_deviceView<B> &fv,
                 std::array<tUDof<1>::t_deviceView<B>, nVarsFixed> &u,
                 std::array<tUGrad<1, 3>::t_deviceView<B>, nVarsFixed> &u_grad,
                 int nIter,
                 const t_jsonconfig &settings);
    };

    template <DeviceBackend B, int nVarsFixed>
    void finiteVolumeCellOpTest_SOA_ver0_main(
        FiniteVolume &fv,
        std::array<tUDof<1>, nVarsFixed> &u,
        std::array<tUGrad<1, 3>, nVarsFixed> &u_grad,
        int nIter,
        const t_jsonconfig &settings)
    {
        auto fv_view = fv.deviceView<B>();

        std::array<tUDof<1>::t_deviceView<B>, nVarsFixed> u_view;
        for (int i = 0; i < nVarsFixed; i++)
            u_view[i] = u[i].template deviceView<B>();
        std::array<tUGrad<1, 3>::t_deviceView<B>, nVarsFixed> u_grad_view;
        for (int i = 0; i < nVarsFixed; i++)
            u_grad_view[i] = u_grad[i].template deviceView<B>();

        finiteVolumeCellOpTest_SOA_ver0_entry<B, nVarsFixed>().run(fv_view, u_view, u_grad_view, nIter, settings);
    }
}