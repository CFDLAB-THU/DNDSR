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

    class EvaluatorConfig
    {
        t_jsonconfig config_data;

    public:
        EvaluatorConfig();

        void valid_patch_keys(const t_jsonconfig &config_in);

        void merge_patch(const t_jsonconfig &config_in)
        {
            valid_patch_keys(config_in);
            config_data.merge_patch(config_in);
        }

        const t_jsonconfig &config() const { return config_data; }
    };

    class Evaluator
    {
    public:
        using t_self = Evaluator;
        using t_fv = ssp<CFV::FiniteVolume>;
        t_fv fv;
        using t_bcHandler = ssp<BCHandler>;
        t_bcHandler bcHandler;
        using t_physics = ssp<Physics>;
        t_physics physics;

        using t_config = EvaluatorConfig;

        t_config config;

        const t_jsonconfig &get_config() { return config.config(); }

        template <typename Tc>
        void set_config(Tc &&config_in)
        {
            config.merge_patch(config_in);
        }

        // internal buffers
        TUDof u_face_bufferL;
        TUDof u_face_bufferR;
        std::vector<TUScalar> uScalar_face_bufferL;
        std::vector<TUScalar> uScalar_face_bufferR;

        void BuildFaceBufferDof(TUDof &u)
        {
            // TODO: upgrade to matching face quad instead
            DNDS_check_throw(fv);
            bool father_changed{false}, son_changed{false};
            if (!u.father)
                u.father = make_ssp<decltype(u.father)::element_type>(ObjName{"EulerPEvaluator::BuildFaceBufferDof::u.father"}, fv->mesh->getMPI()), father_changed = true;
            if (u.father->Size() != fv->mesh->NumFace())
                u.father->Resize(fv->mesh->NumFace(), nVarsFlow, 1), father_changed = true;

            if (!u.son)
                u.son = make_ssp<decltype(u.son)::element_type>(ObjName{"EulerPEvaluator::BuildFaceBufferDof::u.son"}, fv->mesh->getMPI()), son_changed = true;
            if (u.son->Size() != fv->mesh->NumFaceGhost())
                u.son->Resize(fv->mesh->NumFaceGhost(), nVarsFlow, 1), son_changed = true;
            auto B = fv->device();
            if (B != DeviceBackend::Unknown && father_changed)
                u.father->to_device(B);
            if (B != DeviceBackend::Unknown && son_changed)
                u.son->to_device(B);
        }

        void BuildFaceBufferDofScalar(TUScalar &u)
        {
            // TODO: upgrade to matching face quad instead
            DNDS_check_throw(fv);
            bool father_changed{false}, son_changed{false};
            if (!u.father)
                u.father = make_ssp<decltype(u.father)::element_type>(ObjName{"EulerPEvaluator::BuildFaceBufferDofScalar::u.father"}, fv->mesh->getMPI()), father_changed = true;
            if (u.father->Size() != fv->mesh->NumFace())
                u.father->Resize(fv->mesh->NumFace(), 1, 1), father_changed = true;

            if (!u.son)
                u.son = make_ssp<decltype(u.son)::element_type>(ObjName{"EulerPEvaluator::BuildFaceBufferDofScalar::u.son"}, fv->mesh->getMPI()), son_changed = true;
            if (u.son->Size() != fv->mesh->NumFaceGhost())
                u.son->Resize(fv->mesh->NumFaceGhost(), 1, 1), son_changed = true;
            auto B = fv->device();
            // because ArrayPair's to_device is father and son to device:
            if (B != DeviceBackend::Unknown && father_changed)
                u.father->to_device(B);
            if (B != DeviceBackend::Unknown && son_changed)
                u.son->to_device(B);
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

        static auto device_array_list_Buffer()
        {
            return std::make_tuple(
                DNDS_MAKE_1_MEMBER_PTR_SELF(u_face_bufferL),
                DNDS_MAKE_1_MEMBER_PTR_SELF(u_face_bufferR),
                DNDS_MAKE_1_MEMBER_PTR_SELF(uScalar_face_bufferL),
                DNDS_MAKE_1_MEMBER_PTR_SELF(uScalar_face_bufferR));
        }

        Evaluator(
            t_fv n_fv,
            t_bcHandler n_bcHandler,
            t_physics n_physics) : fv(n_fv), bcHandler(n_bcHandler), physics(n_physics)
        {
        }

        void PrintDataVTKHDF(std::string fname, std::string series_name,
                             std::vector<ssp<TUScalar>> &arrCellCentScalar,
                             const std::vector<std::string> &arrCellCentScalar_names_in,
                             std::vector<ssp<TUVec>> &arrCellCentVec,
                             const std::vector<std::string> &arrCellCentVec_names_in,
                             std::vector<ssp<TUScalar>> &arrNodeScalar,
                             const std::vector<std::string> &arrNodeScalar_names_in,
                             std::vector<ssp<TUVec>> &arrNodeVec,
                             const std::vector<std::string> &arrNodeVec_names_in,
                             ssp<TUDof> uPrimCell,
                             ssp<TUDof> uPrimNode,
                             double t);

        DeviceBackend device()
        {
            DeviceBackend B = fv->device();
            DeviceBackend B_mesh = fv->mesh->device();
            DeviceBackend B_bcHandler = bcHandler->device();
            DeviceBackend B_physics = physics->device();
            DNDS_check_throw(B == B_mesh);
            DNDS_check_throw(B == B_bcHandler);
            DNDS_check_throw(B == B_physics);

            return B;
        }

        template <DeviceBackend B>
        struct t_deviceView
        {
            ssp<CFV::FiniteVolume> fv;
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
                static_assert(std::is_trivially_copyable_v<remove_cvref_t<decltype(fv->deviceView<B>())>>);
                return {fv->deviceView<B>(), bcHandler, physics};
            }
        };

        template <DeviceBackend B>
        t_deviceView<B> deviceView()
        {
            DeviceBackend B_fv = fv->device();
            DNDS_check_throw_info(B_fv == B || (B == DeviceBackend::Host && B_fv == DeviceBackend::Unknown),
                                  fmt::format("B_fv is  {} ", device_backend_name(B_fv)) +
                                      fmt::format("B is  {} ", device_backend_name(B)));

            return t_deviceView<B>{
                fv,
                bcHandler->deviceView<B>(),
                physics->deviceView<B>()};
        }

        void to_host()
        {
            fv->to_host();
            bcHandler->to_host();
            physics->to_host();

            u_face_bufferL.to_host();
            u_face_bufferR.to_host();
            for (auto &a : uScalar_face_bufferL)
                a.to_host();
            for (auto &a : uScalar_face_bufferR)
                a.to_host();
        }

        void to_device(DeviceBackend B)
        {
            fv->to_device(B);
            bcHandler->to_device(B);
            physics->to_device(B);
            u_face_bufferL.to_device(B);
            u_face_bufferR.to_device(B);
            for (auto &a : uScalar_face_bufferL)
                a.to_device(B);
            for (auto &a : uScalar_face_bufferR)
                a.to_device(B);
        }

        /****************************** */

        template <class TPair>
        void checkValidUDof(
            const ssp<TPair> &u,
            const std::string &name = "unknown",
            int varloc = 0,
            bool includeSon = true,
            DeviceBackend B = DeviceBackend::Unknown,
            bool optional = false)
        {
            std::string emit_info = fmt::format("{} varloc {} check failure", name, varloc);
            if (optional && !u)
                return;

            DNDS_check_throw_info(u, emit_info);
            DNDS_check_throw_info(u->father, emit_info);
            if (includeSon)
                DNDS_check_throw_info(u->son, emit_info);
            DNDS_check_throw_info(u->father->device() == B, emit_info);
            if (includeSon)
                DNDS_check_throw_info(u->son->device() == B, emit_info);
            if (varloc == -1)
            {
            }
            else if (varloc == 0)
            {
                DNDS_check_throw_info(u->father->Size() == fv->mesh->NumCell(), emit_info);
                if (includeSon)
                    DNDS_check_throw_info(u->son->Size() == fv->mesh->NumCellGhost(), emit_info);
            }
            else if (varloc == 1)
            {
                DNDS_check_throw_info(u->father->Size() == fv->mesh->NumFace(), emit_info);
                if (includeSon)
                    DNDS_check_throw_info(u->son->Size() == fv->mesh->NumFaceGhost(), emit_info);
            }
            else if (varloc == 2)
            {
                DNDS_check_throw_info(u->father->Size() == fv->mesh->NumNode(), emit_info);
                if (includeSon)
                    DNDS_check_throw_info(u->son->Size() == fv->mesh->NumNodeGhost(), emit_info);
            }
            else
                DNDS_check_throw_info(false, "varloc should be -1, 0, 1 or 2");
        }

        struct RecGradient_Arg : public EvaluatorArgBase<RecGradient_Arg>
        {
            ssp<TUDof> u;
            ssp<TUGrad> uGrad;
            // out
            std::vector<ssp<TUScalar>> uScalar;
            std::vector<ssp<TUScalarGrad>> uScalarGrad;

            using t_self = RecGradient_Arg;

            static auto member_list()
            {
                return std::make_tuple(
                    DNDS_MAKE_1_MEMBER_PTR_SELF(u),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uGrad),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalar),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarGrad));
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

        struct Cons2PrimMu_Arg : public EvaluatorArgBase<Cons2PrimMu_Arg>
        {
            using t_self = Cons2PrimMu_Arg;

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
            ssp<TUScalar> gamma;
            ssp<TUScalar> mu;
            std::vector<ssp<TUScalar>> muComp;

            static auto member_list()
            {
                return std::make_tuple(
                    DNDS_MAKE_1_MEMBER_PTR_SELF(u),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uGrad),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalar),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarGrad),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uPrim),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uGradPrim),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarPrim),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarGradPrim),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(p),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(T),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(a),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(gamma),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(mu),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(muComp));
            }

            void Validate(Evaluator &self)
            {
                DeviceBackend B = self.device();
                using namespace std::string_literals;
                int varloc = -1;
                // TODO: improve to check if all sizes are compatible (varloc = -1 does not check size now)

                auto validate_member = [&](std::string name, auto &v)
                {
                    if constexpr (is_ssp_v<remove_cvref_t<decltype(v)>>)
                    {
                        self.checkValidUDof(v, name,
                                            varloc, true, B);
                    }
                    else
                    {
                        for (size_t i = 0; i < v.size(); i++)
                            self.checkValidUDof(v[i], name + "_"s + std::to_string(i),
                                                varloc, true, B);
                    }
                };
                for_each_member_ptr_list(*this,
                                         t_self::member_list(),
                                         validate_member);
            }
        };

        void Cons2PrimMu(Cons2PrimMu_Arg &arg);

        struct Cons2Prim_Arg : public EvaluatorArgBase<Cons2Prim_Arg>
        {
            using t_self = Cons2Prim_Arg;

            ssp<TUDof> u;
            std::vector<ssp<TUScalar>> uScalar;
            // out
            ssp<TUDof> uPrim;
            std::vector<ssp<TUScalar>> uScalarPrim;
            ssp<TUScalar> p;
            ssp<TUScalar> T;
            ssp<TUScalar> a;
            ssp<TUScalar> gamma;

            static auto member_list()
            {
                return std::make_tuple(
                    DNDS_MAKE_1_MEMBER_PTR_SELF(u),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalar),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uPrim),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarPrim),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(p),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(T),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(a),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(gamma));
            }
            void Validate(Evaluator &self)
            {
                DeviceBackend B = self.device();
                using namespace std::string_literals;
                int varloc = -1;

                auto validate_member = [&](std::string name, auto &v)
                {
                    if constexpr (is_ssp_v<remove_cvref_t<decltype(v)>>)
                    {
                        self.checkValidUDof(v, name,
                                            varloc, true, B);
                    }
                    else
                    {
                        for (size_t i = 0; i < v.size(); i++)
                            self.checkValidUDof(v[i], name + "_"s + std::to_string(i),
                                                varloc, true, B);
                    }
                };
                for_each_member_ptr_list(*this,
                                         t_self::member_list(),
                                         validate_member);
            }
        };

        void Cons2Prim(Cons2Prim_Arg &arg);

        struct EstEigenDt_Arg : public EvaluatorArgBase<EstEigenDt_Arg>
        {
            using t_self = EstEigenDt_Arg;
            ssp<TUDof> u;
            ssp<TUScalar> muCell;
            ssp<TUScalar> aCell;
            ssp<TUScalarGrad> faceLamEst;
            ssp<TUScalar> faceLamVisEst;
            ssp<TUScalar> deltaLamFace;
            ssp<TUScalar> deltaLamCell;
            ssp<TUScalar> dt;

            static auto member_list()
            {
                return std::make_tuple(
                    DNDS_MAKE_1_MEMBER_PTR_SELF(u),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(muCell),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(aCell),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(faceLamEst),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(faceLamVisEst),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(deltaLamFace),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(deltaLamCell),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(dt));
            }
            void Validate(Evaluator &self)
            {
                DeviceBackend B = self.device();

                auto validate_member = [&](std::string name, auto &v)
                {
                    int varloc = 0;
                    if (std::set<std::string>{
                            "faceLamEst",
                            "faceLamVisEst",
                            "deltaLamFace",
                        }
                            .count(name))
                        varloc = 1;
                    self.checkValidUDof(v, name, varloc, true, B);
                };
                for_each_member_ptr_list(*this,
                                         t_self::member_list(),
                                         validate_member);
            }
        };

        void EstEigenDt(EstEigenDt_Arg &arg);

        struct RecFace2nd_Arg : public EvaluatorArgBase<RecFace2nd_Arg>
        {
            using t_self = RecFace2nd_Arg;
            ssp<TUDof> u;
            ssp<TUGrad> uGrad;
            std::vector<ssp<TUScalar>> uScalar;
            std::vector<ssp<TUScalarGrad>> uScalarGrad;
            // out
            ssp<TUDof> uFL;
            ssp<TUDof> uFR;
            ssp<TUGrad> uGradFF;
            std::vector<ssp<TUScalar>> uScalarFL;
            std::vector<ssp<TUScalar>> uScalarFR;
            std::vector<ssp<TUScalarGrad>> uScalarGradFF;

            static auto member_list()
            {
                return std::make_tuple(
                    DNDS_MAKE_1_MEMBER_PTR_SELF(u),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uGrad),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalar),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarGrad),
                    // out
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uFL),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uFR),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uGradFF),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarFL),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarFR),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarGradFF));
            }
            void Validate(Evaluator &self)
            {
                DeviceBackend B = self.device();
                using namespace std::string_literals;

                auto validate_member = [&](std::string name, auto &v)
                {
                    int varloc = 0;
                    auto name_last2 = name.substr(std::max(name.size(), 2ul) - 2, 2);
                    if (name_last2 == "FL" || name_last2 == "FR" || name_last2 == "FF")
                        varloc = 1;
                    if constexpr (is_ssp_v<remove_cvref_t<decltype(v)>>)
                        self.checkValidUDof(v, name,
                                            varloc, true, B);

                    else

                        for (size_t i = 0; i < v.size(); i++)
                            self.checkValidUDof(v[i], name + "_"s + std::to_string(i),
                                                varloc, true, B);
                };
                for_each_member_ptr_list(*this,
                                         t_self::member_list(),
                                         validate_member);
            }
        };

        void RecFace2nd(RecFace2nd_Arg &arg);

        struct Flux2nd_Arg : public EvaluatorArgBase<Flux2nd_Arg>
        {
            using t_self = Flux2nd_Arg;
            ssp<TUDof> u;
            ssp<TUGrad> uGrad;
            std::vector<ssp<TUScalar>> uScalar;
            std::vector<ssp<TUScalarGrad>> uScalarGrad;
            ssp<TUDof> uPrim;
            ssp<TUGrad> uGradPrim;
            std::vector<ssp<TUScalar>> uScalarPrim;
            std::vector<ssp<TUScalarGrad>> uScalarGradPrim;
            ssp<TUScalar> p;
            ssp<TUScalar> T;
            ssp<TUScalar> a;
            ssp<TUScalar> mu;
            std::vector<ssp<TUScalar>> muComp;
            ssp<TUScalar> gamma;
            ssp<TUScalar> deltaLamCell;

            ssp<TUDof> uFL;
            ssp<TUDof> uFR;
            ssp<TUGrad> uGradFF;
            std::vector<ssp<TUScalar>> uScalarFL;
            std::vector<ssp<TUScalar>> uScalarFR;
            std::vector<ssp<TUScalarGrad>> uScalarGradFF;

            ssp<TUScalar> pFL;
            ssp<TUScalar> pFR;

            // out
            ssp<TUDof> fluxFF;
            std::vector<ssp<TUScalar>> fluxScalarFF;
            // out added
            ssp<TUDof> rhs;
            std::vector<ssp<TUScalar>> rhsScalar;

            static auto member_list()
            {
                return std::make_tuple(
                    DNDS_MAKE_1_MEMBER_PTR_SELF(u),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uGrad),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalar),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarGrad),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uPrim),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uGradPrim),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarPrim),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarGradPrim),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(p),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(T),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(a),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(gamma),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(mu),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(muComp),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(deltaLamCell),

                    DNDS_MAKE_1_MEMBER_PTR_SELF(uFL),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uFR),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uGradFF),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarFL),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarFR),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(uScalarGradFF),

                    DNDS_MAKE_1_MEMBER_PTR_SELF(pFL),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(pFR),
                    // out
                    DNDS_MAKE_1_MEMBER_PTR_SELF(fluxFF),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(fluxScalarFF),
                    // out added
                    DNDS_MAKE_1_MEMBER_PTR_SELF(rhs),
                    DNDS_MAKE_1_MEMBER_PTR_SELF(rhsScalar));
            }
            void Validate(Evaluator &self)
            {
                DeviceBackend B = self.device();
                using namespace std::string_literals;

                auto validate_member = [&](std::string name, auto &v)
                {
                    int varloc = 0;
                    auto name_last2 = name.substr(std::max(name.size(), 2ul) - 2, 2);
                    if (name_last2 == "FL" || name_last2 == "FR" || name_last2 == "FF")
                        varloc = 1;
                    if constexpr (is_ssp_v<remove_cvref_t<decltype(v)>>)
                        self.checkValidUDof(v, name,
                                            varloc, true, B);

                    else

                        for (size_t i = 0; i < v.size(); i++)
                            self.checkValidUDof(v[i], name + "_"s + std::to_string(i),
                                                varloc, true, B);
                };
                for_each_member_ptr_list(*this,
                                         t_self::member_list(),
                                         validate_member);
            }
        };

        void Flux2nd(Flux2nd_Arg &arg);
    };

    /********************************************************************************** */

    /********************************************************************************** */

}