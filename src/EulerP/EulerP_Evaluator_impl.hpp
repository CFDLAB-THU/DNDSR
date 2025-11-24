#pragma once

#include "EulerP_Evaluator.hpp"
#include <memory>
#include <type_traits>

namespace DNDS::EulerP
{
#define DNDS_EULERP_IMPL_ARG_GET_REF(member) auto &member = arg.member;
#define DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(member) auto &member = arg.portable.member;

#define DNDS_EULERP_IMPL_ARG_CTOR_INIT_SELF()  \
    self(self_),                               \
        this_v(self.template deviceView<B>()), \
        self_view(this_v)
#define DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(member) \
    member(arg.member->template deviceView<B>())
#define DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(member, member_v)    \
    member_v(arg.member.size(), [&](int i)                            \
             { return arg.member.at(i)->template deviceView<B>(); }), \
        member((member_v).deviceView())

#define DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(member)  \
    {                                                           \
        portable.member = arg.member->template deviceView<B>(); \
    }
#define DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR_BUF(member, buf_name) \
    {                                                                        \
        portable.member = (buf_name).template deviceView<B>();               \
    }
#define DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(member, member_v)     \
                                                                                \
    {                                                                           \
        member_v = std::make_unique<typename decltype(member_v)::element_type>( \
            arg.member.size(), [&](int i)                                       \
            { return arg.member.at(i)->template deviceView<B>(); });            \
        portable.member = (member_v)->deviceView();                             \
    }

#define DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR_BUF(member, member_v, buf_name) \
                                                                                          \
    {                                                                                     \
        member_v = std::make_unique<typename decltype(member_v)::element_type>(           \
            (buf_name).size(), [&](int i)                                                 \
            { return (buf_name).at(i).template deviceView<B>(); });                       \
        portable.member = (member_v)->deviceView();                                       \
    }

    template <DeviceBackend B>
    struct Evaluator_impl
    {
        using t_Scalar_deviceViewVector_sup = std::unique_ptr<deviceViewVector<TUScalar::t_deviceView<B>, B>>;
        using t_ScalarGrad_deviceViewVector_sup = std::unique_ptr<deviceViewVector<TUScalarGrad::t_deviceView<B>, B>>;
        struct RecGradient_Arg
        {
            Evaluator &self;
            Evaluator::t_deviceView<B> this_v; //! must keep this alive
            EvaluatorDeviceView<B> self_view;

            struct Portable
            {
                // buffer
                TUDof::t_deviceView<B> faceBCBuffer;
                vector_DeviceView<B, TUScalar::t_deviceView<B>> faceBCScalarBuffer;

                TUDof::t_deviceView<B> u;
                TUGrad::t_deviceView<B> uGrad;
                vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalar;
                vector_DeviceView<B, TUScalarGrad::t_deviceView<B>> uScalarGrad;
            } portable;
            static_assert(std::is_trivially_copyable_v<Portable>);

            t_Scalar_deviceViewVector_sup faceBCScalarBuffer_v;
            // out
            t_Scalar_deviceViewVector_sup uScalar_v;
            t_ScalarGrad_deviceViewVector_sup uScalarGrad_v;

            RecGradient_Arg(Evaluator &self_, Evaluator::RecGradient_Arg &arg)
                : DNDS_EULERP_IMPL_ARG_CTOR_INIT_SELF()
            {
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR_BUF(faceBCBuffer, self.u_face_bufferL)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR_BUF(faceBCScalarBuffer, faceBCScalarBuffer_v, self.uScalar_face_bufferL)

                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(u)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(uGrad)

                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(uScalar, uScalar_v)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(uScalarGrad, uScalarGrad_v)
            }
        };
        static void RecGradient_GGRec(RecGradient_Arg &arg);

        static void RecGradient_BarthLimiter(RecGradient_Arg &arg);

        struct Cons2PrimMu_Arg
        {
            Evaluator &self;
            Evaluator::t_deviceView<B> this_v; //! must keep this alive
            EvaluatorDeviceView<B> self_view;

            struct Portable
            {
                TUDof::t_deviceView<B> u;
                TUGrad::t_deviceView<B> uGrad;
                vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalar;
                vector_DeviceView<B, TUScalarGrad::t_deviceView<B>> uScalarGrad;
                TUDof::t_deviceView<B> uPrim;
                TUGrad::t_deviceView<B> uGradPrim;
                vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalarPrim;
                vector_DeviceView<B, TUScalarGrad::t_deviceView<B>> uScalarGradPrim;
                TUScalar::t_deviceView<B> p;
                TUScalar::t_deviceView<B> T;
                TUScalar::t_deviceView<B> a;
                TUScalar::t_deviceView<B> gamma;
                TUScalar::t_deviceView<B> mu;
                vector_DeviceView<B, TUScalar::t_deviceView<B>> muComp;
            } portable;
            static_assert(std::is_trivially_copyable_v<Portable>);
            t_Scalar_deviceViewVector_sup uScalar_v;
            t_ScalarGrad_deviceViewVector_sup uScalarGrad_v;
            // out
            t_Scalar_deviceViewVector_sup uScalarPrim_v;
            t_ScalarGrad_deviceViewVector_sup uScalarGradPrim_v;
            t_Scalar_deviceViewVector_sup muComp_v;

            Cons2PrimMu_Arg(Evaluator &self_, Evaluator::Cons2PrimMu_Arg &arg)
                : DNDS_EULERP_IMPL_ARG_CTOR_INIT_SELF()
            //
            {
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(u)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(uGrad)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(uScalar, uScalar_v)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(uScalarGrad, uScalarGrad_v)
                // out
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(uPrim)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(uGradPrim)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(uScalarPrim, uScalarPrim_v)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(uScalarGradPrim, uScalarGradPrim_v)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(p)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(T)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(a)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(gamma)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(mu)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(muComp, muComp_v)
            }
        };

        static void Cons2PrimMu(Cons2PrimMu_Arg &arg);

        struct Cons2Prim_Arg
        {
            Evaluator &self;
            Evaluator::t_deviceView<B> this_v; //! must keep this alive
            EvaluatorDeviceView<B> self_view;

            TUDof::t_deviceView<B> u;
            deviceViewVector<TUScalar::t_deviceView<B>, B> uScalar_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalar;
            // out
            TUDof::t_deviceView<B> uPrim;
            deviceViewVector<TUScalar::t_deviceView<B>, B> uScalarPrim_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalarPrim;
            TUScalar::t_deviceView<B> p;
            TUScalar::t_deviceView<B> T;
            TUScalar::t_deviceView<B> a;
            TUScalar::t_deviceView<B> gamma;

            Cons2Prim_Arg(Evaluator &self_, Evaluator::Cons2Prim_Arg &arg)
                : DNDS_EULERP_IMPL_ARG_CTOR_INIT_SELF(),
                  //
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(u),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalar, uScalar_v),
                  // out
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uPrim),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarPrim, uScalarPrim_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(p),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(T),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(a),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(gamma)
            {
            }
        };

        static void Cons2Prim(Cons2Prim_Arg &arg);

        struct EstEigenDt_Arg
        {
            Evaluator &self;
            Evaluator::t_deviceView<B> this_v; //! must keep this alive
            EvaluatorDeviceView<B> self_view;

            TUDof::t_deviceView<B> u;
            TUScalar::t_deviceView<B> muCell;
            TUScalar::t_deviceView<B> aCell;
            TUScalarGrad::t_deviceView<B> faceLamEst;
            TUScalar::t_deviceView<B> faceLamVisEst;
            TUScalar::t_deviceView<B> deltaLamFace;
            TUScalar::t_deviceView<B> deltaLamCell;
            TUScalar::t_deviceView<B> dt;

            EstEigenDt_Arg(Evaluator &self_, Evaluator::EstEigenDt_Arg &arg)
                : DNDS_EULERP_IMPL_ARG_CTOR_INIT_SELF(),
                  //
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(u),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(muCell),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(aCell),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(faceLamEst),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(faceLamVisEst),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(deltaLamFace),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(deltaLamCell),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(dt)
            {
            }
        };

        static void EstEigenDt_GetFaceLam(EstEigenDt_Arg &arg);

        static void EstEigenDt_FaceLam2CellDt(EstEigenDt_Arg &arg);

        struct RecFace2nd_Arg
        {
            Evaluator &self;
            Evaluator::t_deviceView<B> this_v; //! must keep this alive
            EvaluatorDeviceView<B> self_view;

            TUDof::t_deviceView<B> u;
            TUGrad::t_deviceView<B> uGrad;
            deviceViewVector<TUScalar::t_deviceView<B>, B> uScalar_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalar;
            deviceViewVector<TUScalarGrad::t_deviceView<B>, B> uScalarGrad_v;
            vector_DeviceView<B, TUScalarGrad::t_deviceView<B>> uScalarGrad;
            // out
            TUDof::t_deviceView<B> uFL;
            TUDof::t_deviceView<B> uFR;
            TUGrad::t_deviceView<B> uGradFF;
            deviceViewVector<TUScalar::t_deviceView<B>, B> uScalarFL_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalarFL;
            deviceViewVector<TUScalar::t_deviceView<B>, B> uScalarFR_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalarFR;
            deviceViewVector<TUScalarGrad::t_deviceView<B>, B> uScalarGradFF_v;
            vector_DeviceView<B, TUScalarGrad::t_deviceView<B>> uScalarGradFF;

            RecFace2nd_Arg(Evaluator &self_, Evaluator::RecFace2nd_Arg &arg)
                : DNDS_EULERP_IMPL_ARG_CTOR_INIT_SELF(),
                  //
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(u),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uGrad),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalar, uScalar_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarGrad, uScalarGrad_v),
                  // out
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uFL),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uFR),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uGradFF),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarFL, uScalarFL_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarFR, uScalarFR_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarGradFF, uScalarGradFF_v)
            {
            }
        };

        static void RecFace2nd(RecFace2nd_Arg &arg);

        struct Flux2nd_Arg
        {
            Evaluator &self;
            Evaluator::t_deviceView<B> this_v; //! must keep this alive
            EvaluatorDeviceView<B> self_view;

            TUDof::t_deviceView<B> u;
            TUGrad::t_deviceView<B> uGrad;
            deviceViewVector<TUScalar::t_deviceView<B>, B> uScalar_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalar;
            deviceViewVector<TUScalarGrad::t_deviceView<B>, B> uScalarGrad_v;
            vector_DeviceView<B, TUScalarGrad::t_deviceView<B>> uScalarGrad;

            TUDof::t_deviceView<B> uPrim;
            TUGrad::t_deviceView<B> uGradPrim;
            deviceViewVector<TUScalar::t_deviceView<B>, B> uScalarPrim_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalarPrim;
            deviceViewVector<TUScalarGrad::t_deviceView<B>, B> uScalarGradPrim_v;
            vector_DeviceView<B, TUScalarGrad::t_deviceView<B>> uScalarGradPrim;
            TUScalar::t_deviceView<B> p;
            TUScalar::t_deviceView<B> T;
            TUScalar::t_deviceView<B> a;
            TUScalar::t_deviceView<B> gamma;
            TUScalar::t_deviceView<B> mu;
            deviceViewVector<TUScalar::t_deviceView<B>, B> muComp_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> muComp;

            TUDof::t_deviceView<B> uFL;
            TUDof::t_deviceView<B> uFR;
            TUGrad::t_deviceView<B> uGradFF;
            deviceViewVector<TUScalar::t_deviceView<B>, B> uScalarFL_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalarFL;
            deviceViewVector<TUScalar::t_deviceView<B>, B> uScalarFR_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalarFR;
            deviceViewVector<TUScalarGrad::t_deviceView<B>, B> uScalarGradFF_v;
            vector_DeviceView<B, TUScalarGrad::t_deviceView<B>> uScalarGradFF;

            TUScalar::t_deviceView<B> pFL;
            TUScalar::t_deviceView<B> pFR;

            TUScalar::t_deviceView<B> deltaLamFaceFF;

            // out
            TUDof::t_deviceView<B> fluxFF;
            deviceViewVector<TUScalar::t_deviceView<B>, B> fluxScalarFF_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> fluxScalarFF;
            // out added
            TUDof::t_deviceView<B> rhs;
            deviceViewVector<TUScalar::t_deviceView<B>, B> rhsScalar_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> rhsScalar;

            Flux2nd_Arg(Evaluator &self_, Evaluator::Flux2nd_Arg &arg)
                : DNDS_EULERP_IMPL_ARG_CTOR_INIT_SELF(),
                  //
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(u),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uGrad),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalar, uScalar_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarGrad, uScalarGrad_v),

                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uPrim),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uGradPrim),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarPrim, uScalarPrim_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarGradPrim, uScalarGradPrim_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(p),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(T),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(a),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(gamma),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(mu),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(muComp, muComp_v),
                  //
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uFL),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uFR),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uGradFF),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarFL, uScalarFL_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarFR, uScalarFR_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarGradFF, uScalarGradFF_v),

                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(pFL),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(pFR),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(deltaLamFaceFF),
                  // out
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(fluxFF),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(fluxScalarFF, fluxScalarFF_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(rhs),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(rhsScalar, rhsScalar_v)
            {
            }
        };

        static void Flux2nd(Flux2nd_Arg &arg);
    };
}