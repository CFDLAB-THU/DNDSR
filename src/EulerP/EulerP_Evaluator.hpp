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
#include "DNDS/DeviceStorageHelper.hpp"
#include "DNDS/ObjectUtils.hpp"

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
            std::string emit_info = fmt::format("{} varloc {} check failure", name, varloc);
            DNDS_assert_info(u, emit_info);
            DNDS_assert_info(u->father, emit_info);
            if (includeSon)
                DNDS_assert_info(u->son, emit_info);
            DNDS_assert_info(u->father->device() == B, emit_info);
            if (includeSon)
                DNDS_assert_info(u->son->device() == B, emit_info);
            if (varloc == -1)
            {
            }
            else if (varloc == 0)
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
                DNDS_assert_info(false, "varloc should be -1, 0, 1 or 2");
        }

        struct RecGradient_Arg
        {
            ssp<TUDof> u;
            ssp<TUGrad> uGrad;
            // out
            std::vector<ssp<TUScalar>> uScalar;
            std::vector<ssp<TUScalarGrad>> uScalarGrad;

            auto member_list()
            {
                return std::make_tuple(
                    DNDS_MAKE_1_MEMBER_REF(u),
                    DNDS_MAKE_1_MEMBER_REF(uGrad),
                    DNDS_MAKE_1_MEMBER_REF(uScalar),
                    DNDS_MAKE_1_MEMBER_REF(uScalarGrad));
            }

            void Validate(Evaluator &self)
            {
                DeviceBackend B = self.device();
                self.checkValidUDof(u, "u", 0, true, B);
                self.checkValidUDof(uGrad, "uGrad", 0, true, B);
                for (size_t i = 0; i < uScalar.size(); i++)
                    self.checkValidUDof(uScalar[i], "uScalar_" + std::to_string(i), 0, true, B);
                for (size_t i = 0; i < uScalarGrad.size(); i++)
                    self.checkValidUDof(uScalarGrad[i], "uScalarGrad_" + std::to_string(i), 0, true, B);
            }
        };

        void RecGradient(RecGradient_Arg &arg);

        struct Cons2PrimMu_Arg
        {
            ssp<TUDof> u;
            ssp<TUGrad> uGrad;
            std::vector<ssp<TUScalar>> uScalar;
            std::vector<ssp<TUScalarGrad>> uScalarGrad;
            // out
            ssp<TUDof> uPrim;
            ssp<TUGrad> uGradPrim;
            std::vector<ssp<TUScalar>> uScalarPrim;
            std::vector<ssp<TUScalarGrad>> uScalarGradPrim;
            ssp<TUScalar> p;
            ssp<TUScalar> T;
            ssp<TUScalar> a;
            ssp<TUScalar> mu;
            std::vector<ssp<TUScalar>> muComp;

            auto member_list()
            {
                return std::make_tuple(
                    DNDS_MAKE_1_MEMBER_REF(u),
                    DNDS_MAKE_1_MEMBER_REF(uGrad),
                    DNDS_MAKE_1_MEMBER_REF(uScalar),
                    DNDS_MAKE_1_MEMBER_REF(uScalarGrad),
                    DNDS_MAKE_1_MEMBER_REF(uPrim),
                    DNDS_MAKE_1_MEMBER_REF(uGradPrim),
                    DNDS_MAKE_1_MEMBER_REF(uScalarPrim),
                    DNDS_MAKE_1_MEMBER_REF(uScalarGradPrim),
                    DNDS_MAKE_1_MEMBER_REF(p),
                    DNDS_MAKE_1_MEMBER_REF(T),
                    DNDS_MAKE_1_MEMBER_REF(a),
                    DNDS_MAKE_1_MEMBER_REF(mu),
                    DNDS_MAKE_1_MEMBER_REF(muComp));
            }

            void Validate(Evaluator &self)
            {
                DeviceBackend B = self.device();
                using namespace std::string_literals;
                int varloc = -1;

                auto validate_member = [&](auto &v)
                {
                    if constexpr (is_ssp_v<remove_cvref_t<decltype(v.ref)>>)
                    {
                        self.checkValidUDof(v.ref, v.name,
                                            varloc, true, B);
                    }
                    else
                    {
                        for (size_t i = 0; i < v.ref.size(); i++)
                            self.checkValidUDof(v.ref[i], v.name + "_"s + std::to_string(i),
                                                varloc, true, B);
                    }
                };
                for_each_member_list(
                    this->member_list(),
                    validate_member);
            }
        };

        void Cons2PrimMu(Cons2PrimMu_Arg &arg);

        struct EstEigenDt_Arg
        {
            ssp<TUDof> u;
            ssp<TUScalar> muCell;
            ssp<TUScalar> aCell;
            ssp<TUScalarGrad> faceLamEst;
            ssp<TUScalar> faceLamVisEst;
            ssp<TUScalar> deltaLamFace;
            ssp<TUScalar> deltaLamCell;
            ssp<TUScalar> dt;

            auto member_list()
            {
                return std::make_tuple(
                    DNDS_MAKE_1_MEMBER_REF(u),
                    DNDS_MAKE_1_MEMBER_REF(muCell),
                    DNDS_MAKE_1_MEMBER_REF(aCell),
                    DNDS_MAKE_1_MEMBER_REF(faceLamEst),
                    DNDS_MAKE_1_MEMBER_REF(faceLamVisEst),
                    DNDS_MAKE_1_MEMBER_REF(deltaLamFace),
                    DNDS_MAKE_1_MEMBER_REF(deltaLamCell),
                    DNDS_MAKE_1_MEMBER_REF(dt));
            }

            void Validate(Evaluator &self)
            {
                DeviceBackend B = self.device();

                auto validate_member = [&](auto &v)
                {
                    int varloc = 0;
                    if (std::set<std::string>{
                            "faceLamEst",
                            "faceLamVisEst",
                            "deltaLamFace",
                        }
                            .count(v.name))
                        varloc = 1;
                    self.checkValidUDof(v.ref, v.name,
                                        varloc, true, B);
                };
                for_each_member_list(
                    this->member_list(),
                    validate_member);
            }
        };

        void EstEigenDt(EstEigenDt_Arg &arg);

        struct Rec2nd_Arg
        {
            ssp<TUDof> u;
            ssp<TUGrad> uGrad;
            std::vector<ssp<TUScalar>> uScalar;
            std::vector<ssp<TUScalarGrad>> uScalarGrad;
            // out
            ssp<TUDof> uFL;
            ssp<TUDof> uFR;
            ssp<TUGrad> uGradFL;
            ssp<TUGrad> uGradFR;
            std::vector<ssp<TUScalar>> uScalarFL;
            std::vector<ssp<TUScalar>> uScalarFR;
            std::vector<ssp<TUScalarGrad>> uScalarGradFL;
            std::vector<ssp<TUScalarGrad>> uScalarGradFR;

            auto member_list()
            {
                return std::make_tuple(
                    DNDS_MAKE_1_MEMBER_REF(u),
                    DNDS_MAKE_1_MEMBER_REF(uGrad),
                    DNDS_MAKE_1_MEMBER_REF(uScalar),
                    DNDS_MAKE_1_MEMBER_REF(uScalarGrad),
                    // out
                    DNDS_MAKE_1_MEMBER_REF(uFL),
                    DNDS_MAKE_1_MEMBER_REF(uFR),
                    DNDS_MAKE_1_MEMBER_REF(uGradFL),
                    DNDS_MAKE_1_MEMBER_REF(uGradFR),
                    DNDS_MAKE_1_MEMBER_REF(uScalarFL),
                    DNDS_MAKE_1_MEMBER_REF(uScalarFR),
                    DNDS_MAKE_1_MEMBER_REF(uScalarGradFL),
                    DNDS_MAKE_1_MEMBER_REF(uScalarGradFR));
            }

            void Validate(Evaluator &self)
            {
                DeviceBackend B = self.device();
                using namespace std::string_literals;

                auto validate_member = [&](auto &v)
                {
                    int varloc = 0;
                    std::string name = v.name;
                    auto name_last2 = name.substr(name.size() - 2, 2);
                    if (name_last2 == "FL" || name_last2 == "FR")
                        varloc = 1;
                    if constexpr (is_ssp_v<remove_cvref_t<decltype(v.ref)>>)
                        self.checkValidUDof(v.ref, v.name,
                                            varloc, true, B);

                    else

                        for (size_t i = 0; i < v.ref.size(); i++)
                            self.checkValidUDof(v.ref[i], v.name + "_"s + std::to_string(i),
                                                varloc, true, B);
                };
                for_each_member_list(
                    this->member_list(),
                    validate_member);
            }
        };

        void Rec2nd(Rec2nd_Arg &arg);

        // consider upgrading this to directly correspond to quad_point?
        struct Flux2nd_Arg
        {
            ssp<TUDof> uFL;
            ssp<TUDof> uFR;
            ssp<TUGrad> uGradFL;
            ssp<TUGrad> uGradFR;
            std::vector<ssp<TUScalar>> uScalarFL;
            std::vector<ssp<TUScalar>> uScalarFR;
            std::vector<ssp<TUScalarGrad>> uScalarGradFL;
            std::vector<ssp<TUScalarGrad>> uScalarGradFR;
            ssp<TUScalar> muFF;
            std::vector<ssp<TUScalar>> muCompFF;
            ssp<TUScalar> deltaLamFaceFF;
            // out
            ssp<TUDof> fluxFF;
            std::vector<ssp<TUScalar>> fluxScalarFF;
            // out added
            ssp<TUDof> rhs;
            std::vector<ssp<TUScalar>> rhsScalar;

            auto member_list()
            {
                return std::make_tuple(
                    DNDS_MAKE_1_MEMBER_REF(uFL),
                    DNDS_MAKE_1_MEMBER_REF(uFR),
                    DNDS_MAKE_1_MEMBER_REF(uGradFL),
                    DNDS_MAKE_1_MEMBER_REF(uGradFR),
                    DNDS_MAKE_1_MEMBER_REF(uScalarFL),
                    DNDS_MAKE_1_MEMBER_REF(uScalarFR),
                    DNDS_MAKE_1_MEMBER_REF(uScalarGradFL),
                    DNDS_MAKE_1_MEMBER_REF(uScalarGradFR),
                    DNDS_MAKE_1_MEMBER_REF(muFF),
                    DNDS_MAKE_1_MEMBER_REF(muCompFF),
                    DNDS_MAKE_1_MEMBER_REF(deltaLamFaceFF),
                    // out
                    DNDS_MAKE_1_MEMBER_REF(fluxFF),
                    DNDS_MAKE_1_MEMBER_REF(fluxScalarFF),
                    // out added
                    DNDS_MAKE_1_MEMBER_REF(rhs),
                    DNDS_MAKE_1_MEMBER_REF(rhsScalar));
            }

            void Validate(Evaluator &self)
            {
                DeviceBackend B = self.device();
                using namespace std::string_literals;

                auto validate_member = [&](auto &v)
                {
                    int varloc = 0;
                    std::string name = v.name;
                    auto name_last2 = name.substr(name.size() - 2, 2);
                    if (name_last2 == "FL" || name_last2 == "FR" || name_last2 == "FF")
                        varloc = 1;
                    if constexpr (is_ssp_v<remove_cvref_t<decltype(v.ref)>>)
                        self.checkValidUDof(v.ref, v.name,
                                            varloc, true, B);

                    else

                        for (size_t i = 0; i < v.ref.size(); i++)
                            self.checkValidUDof(v.ref[i], v.name + "_"s + std::to_string(i),
                                                varloc, true, B);
                };
                for_each_member_list(
                    this->member_list(),
                    validate_member);
            }
        };

        void Flux2nd(Flux2nd_Arg &arg);
    };

    /********************************************************************************** */

    /********************************************************************************** */

}