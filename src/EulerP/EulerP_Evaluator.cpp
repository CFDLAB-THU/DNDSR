#include "EulerP_Evaluator.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "DNDS/Errors.hpp"
#include "DNDS/DeviceStorageHelper.hpp"
#include "EulerP/EulerP.hpp"
#include "EulerP_Evaluator_impl.hpp"
#include <type_traits>

namespace DNDS::EulerP
{

    EvaluatorConfig::EvaluatorConfig()
    {
        auto &c = this->config_data = t_jsonconfig::object();
        c["threadsPerBlock"] = 128;
        c["pullAllInputArgs"] = true;
    }

    void EvaluatorConfig::valid_patch_keys(const t_jsonconfig &config_in)
    {
        auto validate_keys_run = [](const t_jsonconfig &user, const t_jsonconfig &def, const std::string &path, auto &&validate_keys_runF)
        {
            if (!user.is_object() || !def.is_object())
                return;
            for (const auto &item : user.items())
            {
                const std::string &key = item.key();
                const t_jsonconfig &uval = item.value();
                if (!def.contains(key))
                    throw std::runtime_error(
                        "Invalid configuration key at path '" + path + key + "'");

                const t_jsonconfig &dval = def.at(key);

                // If both sides are objects, recurse
                if (uval.is_object() && dval.is_object())
                    validate_keys_runF(uval, dval, path + key + ".", validate_keys_runF);
            }
        };
        validate_keys_run(config_in, config_data, "", validate_keys_run);
    }

    void Evaluator::RecGradient(
        RecGradient_Arg &arg)
    {
        DeviceBackend B = this->device();
        arg.Validate(*this);
        arg.WaitAllPull(B);

        // DNDS_check_throw(u_face_bufferL.father);
        // DNDS_check_throw(uScalar_face_bufferL.size() >= uScalar.size());
        // for (int i = 0; i < uScalar.size(); i++)
        //     DNDS_check_throw(uScalar_face_bufferL[i].father);
        PrepareFaceBuffer(arg.uScalar.size());

        if (B == DeviceBackend::Host || B == DeviceBackend::Unknown)
        {
            constexpr DeviceBackend B = DeviceBackend::Host;
            Evaluator_impl<B>::RecGradient_Arg impl_arg(*this, arg);

            arg.u->trans.waitPersistentPull();
            for (auto &uS : arg.uScalar)
                uS->trans.waitPersistentPull();

            Evaluator_impl<B>::RecGradient_GGRec(impl_arg);

            Evaluator_impl<B>::RecGradient_BarthLimiter(impl_arg);
        }
#ifdef DNDS_USE_CUDA
        else if (B == DeviceBackend::CUDA)
        {
            constexpr DeviceBackend B = DeviceBackend::CUDA;
            Evaluator_impl<B>::RecGradient_Arg impl_arg(*this, arg);

            arg.u->trans.waitPersistentPull();
            for (auto &uS : arg.uScalar)
                uS->trans.waitPersistentPull();

            Evaluator_impl<B>::RecGradient_GGRec(impl_arg);

            Evaluator_impl<B>::RecGradient_BarthLimiter(impl_arg);
        }
#endif
        else
            DNDS_check_throw(false);
    }

    void Evaluator::Cons2PrimMu(Cons2PrimMu_Arg &arg)
    {
        DeviceBackend B = this->device();
        arg.Validate(*this);
        arg.WaitAllPull(B);

        auto execute = [&](auto b = std::integral_constant<DeviceBackend, DeviceBackend::Host>())
        {
            constexpr DeviceBackend B = decltype(b)::value;

            typename Evaluator_impl<B>::Cons2PrimMu_Arg impl_arg(*this, arg);

            // todo: conditionally disable comm
            // arg.u->trans.waitPersistentPull();
            // for (auto &uS : arg.uScalar)
            //     uS->trans.waitPersistentPull();
            // arg.uGrad->trans.waitPersistentPull();
            // for (auto &uS : arg.uScalarGrad)
            //     uS->trans.waitPersistentPull();

            Evaluator_impl<B>::Cons2PrimMu(impl_arg);

            // todo: conditionally disable comm
            // arg.uPrim->trans.startPersistentPull();
            // for (auto &uS : arg.uScalarPrim)
            //     uS->trans.startPersistentPull();
            // arg.p->trans.startPersistentPull();
            // arg.T->trans.startPersistentPull();
            // arg.a->trans.startPersistentPull();
            // arg.mu->trans.startPersistentPull();
            // for (auto &uS : arg.muComp)
            //     uS->trans.startPersistentPull();
        };

        if (B == DeviceBackend::Host || B == DeviceBackend::Unknown)
        {
            execute(std::integral_constant<DeviceBackend, DeviceBackend::Host>());
        }
#ifdef DNDS_USE_CUDA
        else if (B == DeviceBackend::CUDA)
        {
            execute(std::integral_constant<DeviceBackend, DeviceBackend::CUDA>());
        }
#endif
        else
            DNDS_check_throw(false);
    }

    void Evaluator::Cons2Prim(Cons2Prim_Arg &arg)
    {
        DeviceBackend B = this->device();
        arg.Validate(*this);
        arg.WaitAllPull(B);

        auto execute = [&](auto b)
        {
            constexpr DeviceBackend B = decltype(b)::value;

            typename Evaluator_impl<B>::Cons2Prim_Arg impl_arg(*this, arg);

            // todo: conditionally disable comm
            // arg.u->trans.waitPersistentPull();
            // for (auto &uS : arg.uScalar)
            //     uS->trans.waitPersistentPull();

            Evaluator_impl<B>::Cons2Prim(impl_arg);

            // todo: conditionally disable comm
            // arg.uPrim->trans.startPersistentPull();
            // for (auto &uS : arg.uScalarPrim)
            //     uS->trans.startPersistentPull();
            // arg.p->trans.startPersistentPull();
            // arg.T->trans.startPersistentPull();
            // arg.a->trans.startPersistentPull();
        };

        if (B == DeviceBackend::Host || B == DeviceBackend::Unknown)
        {
            execute(std::integral_constant<DeviceBackend, DeviceBackend::Host>());
        }
#ifdef DNDS_USE_CUDA
        else if (B == DeviceBackend::CUDA)
        {
            execute(std::integral_constant<DeviceBackend, DeviceBackend::CUDA>());
        }
#endif
        else
            DNDS_check_throw(false);
    }

    void Evaluator::EstEigenDt(EstEigenDt_Arg &arg)
    {
        DeviceBackend B = this->device();
        arg.Validate(*this);
        arg.WaitAllPull(B);

        auto execute = [&](auto b = std::integral_constant<DeviceBackend, DeviceBackend::Host>())
        {
            constexpr DeviceBackend B = decltype(b)::value;

            typename Evaluator_impl<B>::EstEigenDt_Arg impl_arg(*this, arg);

            Evaluator_impl<B>::EstEigenDt_GetFaceLam(impl_arg);

            //! this is not needed, as we have ensured that all valid-internal faces have valid state
            // faceLamEst->trans.startPersistentPull();
            // faceLamVisEst->trans.startPersistentPull();
            // deltaLamFace->trans.startPersistentPull();

            // faceLamEst->trans.waitPersistentPull();
            // faceLamVisEst->trans.waitPersistentPull();
            // deltaLamFace->trans.waitPersistentPull();

            Evaluator_impl<B>::EstEigenDt_FaceLam2CellDt(impl_arg);
        };

        if (B == DeviceBackend::Host || B == DeviceBackend::Unknown)
        {
            execute(std::integral_constant<DeviceBackend, DeviceBackend::Host>());
        }
#ifdef DNDS_USE_CUDA
        else if (B == DeviceBackend::CUDA)
        {
            execute(std::integral_constant<DeviceBackend, DeviceBackend::CUDA>());
        }
#endif
        else
            DNDS_check_throw(false);
    }

    void Evaluator::RecFace2nd(RecFace2nd_Arg &arg)
    {
        DeviceBackend B = this->device();
        arg.Validate(*this);
        arg.WaitAllPull(B);
        auto execute = [&](auto b = std::integral_constant<DeviceBackend, DeviceBackend::Host>())
        {
            constexpr DeviceBackend B = decltype(b)::value;

            typename Evaluator_impl<B>::RecFace2nd_Arg impl_arg(*this, arg);

            Evaluator_impl<B>::RecFace2nd(impl_arg);
        };

        if (B == DeviceBackend::Host || B == DeviceBackend::Unknown)
        {
            execute(std::integral_constant<DeviceBackend, DeviceBackend::Host>());
        }
#ifdef DNDS_USE_CUDA
        else if (B == DeviceBackend::CUDA)
        {
            execute(std::integral_constant<DeviceBackend, DeviceBackend::CUDA>());
        }
#endif
        else
            DNDS_check_throw(false);
    }

    void Evaluator::Flux2nd(Flux2nd_Arg &arg)
    {
        DeviceBackend B = this->device();
        arg.Validate(*this);
        PrepareFaceBuffer(arg.uScalar.size());
        arg.WaitAllPull(B);

        auto execute = [&](auto b = std::integral_constant<DeviceBackend, DeviceBackend::Host>())
        {
            constexpr DeviceBackend B = decltype(b)::value;

            typename Evaluator_impl<B>::Flux2nd_Arg impl_arg(*this, arg);

            for (auto &uS : arg.uScalar)
                uS->trans.waitPersistentPull();
            for (auto &uS : arg.uScalarPrim)
                uS->trans.waitPersistentPull();
            for (auto &uS : arg.uScalarGrad)
                uS->trans.waitPersistentPull();
            for (auto &uS : arg.uScalarGradPrim)
                uS->trans.waitPersistentPull();

            Evaluator_impl<B>::Flux2nd(impl_arg);

            // arg.rhs->trans.startPersistentPull();
            // for (auto &uS : arg.rhsScalar)
            //     uS->trans.startPersistentPull();
        };

        if (B == DeviceBackend::Host || B == DeviceBackend::Unknown)
        {
            execute(std::integral_constant<DeviceBackend, DeviceBackend::Host>());
        }
#ifdef DNDS_USE_CUDA
        else if (B == DeviceBackend::CUDA)
        {
            execute(std::integral_constant<DeviceBackend, DeviceBackend::CUDA>());
        }
#endif
        else
            DNDS_check_throw(false);
    }
}