#pragma once

#include "CFV/FiniteVolume.hpp"
#include "EulerP.hpp"
#include "EulerP_BC.hpp"
#include "EulerP_Physics.hpp"

namespace DNDS::EulerP
{

    using TFiniteVolume = CFV::FiniteVolume;

    template <DeviceBackend B>
    class EvaluatorDeviceView
    {
    public:
        using t_fv = TFiniteVolume::t_deviceView<B>;
        t_fv fv;
        using t_bcHandler = BCHandlerDeviceView<B>;
        t_bcHandler bcHandler;
        using t_physics = PhysicsDeviceView<B>;
        t_physics physics;

        DNDS_DEVICE_TRIVIAL_COPY_DEFINE_NO_EMPTY_CTOR(EvaluatorDeviceView, EvaluatorDeviceView)
        DNDS_DEVICE_CALLABLE EvaluatorDeviceView(
            t_fv n_fv,
            t_bcHandler n_bcHandler,
            t_physics n_physics) : fv(n_fv), bcHandler(n_bcHandler), physics(n_physics) {}
    };

    class Evaluator
    {
    public:
        using t_fv = ssp<CFV::FiniteVolume>;
        t_fv fv;
        using t_bcHandler = ssp<BCHandler>;
        t_bcHandler bcHandler;
        using t_physics = ssp<Physics>;
        t_physics physics;

        Evaluator(
            t_fv n_fv,
            t_bcHandler n_bcHandler,
            t_physics n_physics) : fv(n_fv), bcHandler(n_bcHandler), physics(n_physics) {}

        void PrintDataVTKHDF(std::string fname, std::string series_name,
                             std::vector<ssp<TUScalar>> &arrCellCentScalar,
                             const std::vector<std::string> &arrCellCentScalar_names,
                             std::vector<ssp<TUVec>> &arrCellCentVec,
                             const std::vector<std::string> &arrCellCentVec_names,
                             std::vector<ssp<TUScalar>> &arrNodeScalar,
                             const std::vector<std::string> &arrNodeScalar_names,
                             std::vector<ssp<TUVec>> &arrNodeVec,
                             const std::vector<std::string> &arrNodeVec_names,
                             double t)
        {
            MPI_Comm commDup = MPI_COMM_NULL;
            MPI_Comm_dup(fv->mesh->getMPI().comm, &commDup);

            for (auto &arr : arrCellCentScalar)
            {
                DNDS_assert(arr->father);
                DNDS_assert(arr->father->Size() == fv->mesh->NumCell());
            }
            for (auto &arr : arrCellCentVec)
            {
                DNDS_assert(arr->father);
                DNDS_assert(arr->father->Size() == fv->mesh->NumCell());
            }
            for (auto &arr : arrNodeScalar)
            {
                DNDS_assert(arr->father);
                DNDS_assert(arr->father->Size() == fv->mesh->NumNode());
            }
            for (auto &arr : arrNodeVec)
            {
                DNDS_assert(arr->father);
                DNDS_assert(arr->father->Size() == fv->mesh->NumNode());
            }
            DNDS_assert(arrCellCentScalar_names.size() == arrCellCentScalar.size());
            DNDS_assert(arrCellCentVec_names.size() == arrCellCentVec.size());
            DNDS_assert(arrNodeScalar_names.size() == arrNodeScalar.size());
            DNDS_assert(arrNodeVec_names.size() == arrNodeVec.size());
            fv->mesh->PrintParallelVTKHDFDataArray(
                fname,
                series_name,
                arrCellCentScalar.size(),
                arrCellCentVec.size(),
                arrNodeScalar.size(),
                arrNodeVec.size(),
                [&](int i)
                { return arrCellCentScalar_names.at(i); },
                [&](int i, index iC)
                { return arrCellCentScalar.at(i)->father->operator[](iC)(0); },
                [&](int i)
                { return arrCellCentVec_names.at(i); },
                [&](int i, index iC, rowsize iV)
                { return arrCellCentVec.at(i)->father->operator[](iC)(iV); },
                [&](int i)
                { return arrNodeScalar_names.at(i); },
                [&](int i, index iN)
                { return arrNodeScalar.at(i)->father->operator[](iN)(0); },
                [&](int i)
                { return arrNodeVec_names.at(i); },
                [&](int i, index iN, rowsize iV)
                { return arrNodeVec.at(i)->father->operator[](iN)(iV); },
                t,
                commDup);
            MPI_Comm_free(&commDup);
        }
    };
}