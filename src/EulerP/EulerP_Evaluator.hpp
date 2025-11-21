#pragma once

#include "CFV/FiniteVolume.hpp"
#include "DNDS/ArrayDOF.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "DNDS/Errors.hpp"
#include "EulerP.hpp"
#include "EulerP_BC.hpp"
#include "EulerP_Physics.hpp"
#include <vector>

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

        // internal buffers
        TUDof u_face_bufferL;
        TUDof u_face_bufferR;
        std::vector<TUScalar> uScalar_face_bufferL;
        std::vector<TUScalar> uScalar_face_bufferR;

        void BuildFaceBufferDof(TUDof &u)
        {
            // TODO: upgrade to matching face quad instead
            if (!u.father)
                DNDS_MAKE_SSP(u.father, fv->mesh->getMPI());
            if (u.father->Size() != fv->mesh->NumFace())
                u.father->Resize(fv->mesh->NumFace(), nVarsFlow, 1);

            if (!u.son)
                DNDS_MAKE_SSP(u.son, fv->mesh->getMPI());
            if (u.son->Size() != fv->mesh->NumFaceGhost())
                u.son->Resize(fv->mesh->NumFaceGhost(), nVarsFlow, 1);
        }

        void BuildFaceBufferDofScalar(TUScalar &u)
        {
            // TODO: upgrade to matching face quad instead
            if (!u.father)
                DNDS_MAKE_SSP(u.father, fv->mesh->getMPI());
            if (u.father->Size() != fv->mesh->NumFace())
                u.father->Resize(fv->mesh->NumFace(), 1, 1);

            if (!u.son)
                DNDS_MAKE_SSP(u.son, fv->mesh->getMPI());
            if (u.son->Size() != fv->mesh->NumFaceGhost())
                u.son->Resize(fv->mesh->NumFaceGhost(), 1, 1);
        }

        void PrepareFaceBuffer(int nVarsScalar)
        {
            BuildFaceBufferDof(u_face_bufferL);
            BuildFaceBufferDof(u_face_bufferR);

            uScalar_face_bufferL.resize(nVarsScalar);
            uScalar_face_bufferR.resize(nVarsScalar);
            for (auto &uS : uScalar_face_bufferL)
                BuildFaceBufferDofScalar(uS);
            for (auto &uS : uScalar_face_bufferR)
                BuildFaceBufferDofScalar(uS);
        }

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
                size_t_to_signed<int>(arrCellCentScalar.size()),
                size_t_to_signed<int>(arrCellCentVec.size()),
                size_t_to_signed<int>(arrNodeScalar.size()),
                size_t_to_signed<int>(arrNodeVec.size()),
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

        DeviceBackend device()
        {
            DeviceBackend B = fv->device();
            DeviceBackend B_mesh = fv->mesh->device();
            DeviceBackend B_bcHandler = bcHandler->device();
            DeviceBackend B_physics = physics->device();
            DNDS_assert(B == B_mesh);
            DNDS_assert(B == B_bcHandler);
            DNDS_assert(B == B_physics);

            return B;
        }

        template <DeviceBackend B>
        struct t_deviceView
        {
            t_fv::element_type::t_deviceView<B> fv;
            t_bcHandler::element_type::t_deviceView<B> bcHandler;
            t_physics::element_type::t_deviceView<B> physics;

            //! only permit moving to avoid host_device_vector to change
            // also avoids accidentally copying this to device...
            t_deviceView(t_deviceView &&R) noexcept = default;
            t_deviceView(const t_deviceView &R) = delete;
            t_deviceView &operator=(t_deviceView &&R) = delete;
            t_deviceView &operator=(const t_deviceView &R) = delete;

            operator EvaluatorDeviceView<B>() const
            {
                return {fv, bcHandler, physics};
            }
        };

        template <DeviceBackend B>
        t_deviceView<B> deviceView()
        {
            DeviceBackend B_fv = fv->device();
            DNDS_assert(B_fv == B);

            return t_deviceView<B>{
                fv->deviceView<B>(),
                bcHandler->deviceView<B>(),
                physics->deviceView<B>()};
        }

        void to_host()
        {
            fv->to_host();
            bcHandler->to_host();
            physics->to_host();
        }

        void to_device(DeviceBackend B)
        {
            fv->to_device(B);
            bcHandler->to_device(B);
            physics->to_device(B);
        }

        /****************************** */

        template <class TPair>
        void checkValidUDof(
            const ssp<TPair> &u,
            const std::string &name = "unknown",
            int varloc = 0,
            bool includeSon = true,
            DeviceBackend B = DeviceBackend::Unknown)
        {
            std::string emit_info = fmt::format("{}", name);
            DNDS_assert_info(u, emit_info);
            DNDS_assert_info(u->father, emit_info);
            if (includeSon)
                DNDS_assert_info(u->son, emit_info);
            DNDS_assert_info(u->father->device() == B, emit_info);
            if (includeSon)
                DNDS_assert_info(u->son->device() == B, emit_info);
            if (varloc == 0)
            {
                DNDS_assert_info(u->father->Size() == fv->mesh->NumCell(), emit_info);
                if (includeSon)
                    DNDS_assert_info(u->son->Size() == fv->mesh->NumCellGhost(), emit_info);
            }
            else if (varloc == 1)
            {
                DNDS_assert_info(u->father->Size() == fv->mesh->NumFace(), emit_info);
                if (includeSon)
                    DNDS_assert_info(u->son->Size() == fv->mesh->NumFaceGhost(), emit_info);
            }
            else if (varloc == 2)
            {
                DNDS_assert_info(u->father->Size() == fv->mesh->NumNode(), emit_info);
                if (includeSon)
                    DNDS_assert_info(u->son->Size() == fv->mesh->NumNodeGhost(), emit_info);
            }
            else
                DNDS_assert_info(false, "varloc is 0 or 1 or 2");
        }

        void RecGradient(const ssp<TUDof> &u,
                         const ssp<TUGrad> &uGrad,
                         const std::vector<ssp<TUScalar>> &uScalar,
                         const std::vector<ssp<TUScalarGrad>> &uScalarGrad);

        template <DeviceBackend B>
        void RecGradient_impl(const ssp<TUDof> &u,
                              const ssp<TUGrad> &uGrad,
                              const std::vector<ssp<TUScalar>> &uScalar,
                              const std::vector<ssp<TUScalarGrad>> &uScalarGrad);

        void EstEigenDt(const ssp<TUDof> &u,
                        const ssp<TUGrad> &uGrad,
                        const std::vector<ssp<TUScalar>> &uScalar,
                        const std::vector<ssp<TUScalarGrad>> &uScalarGrad,
                        const ssp<TUScalar> &faceLamEst,
                        const ssp<TUScalar> &faceLamVisEst,
                        const ssp<TUScalar> &dt);

        void RHS2nd(const ssp<TUDof> &u,
                    const ssp<TUGrad> &uGrad,
                    const std::vector<ssp<TUScalar>> &uScalar,
                    const std::vector<ssp<TUScalarGrad>> &uScalarGrad,
                    const ssp<TUScalar> &faceLamEst,
                    const ssp<TUScalar> &faceLamVisEst,
                    const ssp<TUDof> &rhs,
                    const ssp<TUDof> &flux_face,
                    const ssp<TUDof> &source);
    };
}