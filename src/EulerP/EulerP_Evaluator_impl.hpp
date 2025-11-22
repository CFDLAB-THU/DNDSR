#pragma once

#include "EulerP_Evaluator.hpp"

namespace DNDS::EulerP
{
#define DNDS_EULERP_IMPL_ARG_GET_REF(member) auto &member = arg.member;

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

    template <DeviceBackend B>
    struct Evaluator_impl
    {
        struct RecGradient_Arg
        {
            Evaluator &self;
            Evaluator::t_deviceView<B> this_v; //! must keep this alive
            EvaluatorDeviceView<B> self_view;

            // buffer
            TUDof::t_deviceView<B> faceBCBuffer;
            deviceViewVector<TUScalar::t_deviceView<B>, B> faceBCScalarBuffer_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> faceBCScalarBuffer;

            // out
            TUDof::t_deviceView<B> u;
            TUGrad::t_deviceView<B> uGrad;
            deviceViewVector<TUScalar::t_deviceView<B>, B> uScalar_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalar;
            deviceViewVector<TUScalarGrad::t_deviceView<B>, B> uScalarGrad_v;
            vector_DeviceView<B, TUScalarGrad::t_deviceView<B>> uScalarGrad;

            RecGradient_Arg(Evaluator &self_, Evaluator::RecGradient_Arg &arg)
                : DNDS_EULERP_IMPL_ARG_CTOR_INIT_SELF(),
                  faceBCBuffer(self.u_face_bufferL.deviceView<B>()),
                  faceBCScalarBuffer_v(self.uScalar_face_bufferL.size(), [&](int i)
                                       { return self.uScalar_face_bufferL.at(i).deviceView<B>(); }),
                  faceBCScalarBuffer(faceBCScalarBuffer_v.deviceView()),
                  //
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(u),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uGrad),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalar, uScalar_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarGrad, uScalarGrad_v)
            {
            }
        };
        static void RecGradient_GGRec(RecGradient_Arg &arg);

        static void RecGradient_BarthLimiter(RecGradient_Arg &arg);

        struct Cons2PrimMu_Arg
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
            TUDof::t_deviceView<B> uPrim;
            TUGrad::t_deviceView<B> uGradPrim;
            deviceViewVector<TUScalar::t_deviceView<B>, B> uScalarPrim_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalarPrim;
            deviceViewVector<TUScalarGrad::t_deviceView<B>, B> uScalarGradPrim_v;
            vector_DeviceView<B, TUScalarGrad::t_deviceView<B>> uScalarGradPrim;
            TUScalar::t_deviceView<B> p;
            TUScalar::t_deviceView<B> T;
            TUScalar::t_deviceView<B> a;
            TUScalar::t_deviceView<B> mu;
            deviceViewVector<TUScalar::t_deviceView<B>, B> muComp_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> muComp;

            Cons2PrimMu_Arg(Evaluator &self_, Evaluator::Cons2PrimMu_Arg &arg)
                : DNDS_EULERP_IMPL_ARG_CTOR_INIT_SELF(),
                  //
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(u),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uGrad),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalar, uScalar_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarGrad, uScalarGrad_v),
                  // out
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uPrim),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uGradPrim),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarPrim, uScalarPrim_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarGradPrim, uScalarGradPrim_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(p),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(T),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(a),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(mu),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(muComp, muComp_v)
            {
            }
        };

        static void Cons2PrimMu(Cons2PrimMu_Arg &arg);

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

        struct Rec2nd_Arg
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
            TUGrad::t_deviceView<B> uGradFL;
            TUGrad::t_deviceView<B> uGradFR;
            deviceViewVector<TUScalar::t_deviceView<B>, B> uScalarFL_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalarFL;
            deviceViewVector<TUScalar::t_deviceView<B>, B> uScalarFR_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalarFR;
            deviceViewVector<TUScalarGrad::t_deviceView<B>, B> uScalarGradFL_v;
            vector_DeviceView<B, TUScalarGrad::t_deviceView<B>> uScalarGradFL;
            deviceViewVector<TUScalarGrad::t_deviceView<B>, B> uScalarGradFR_v;
            vector_DeviceView<B, TUScalarGrad::t_deviceView<B>> uScalarGradFR;

            Rec2nd_Arg(Evaluator &self_, Evaluator::Rec2nd_Arg &arg)
                : DNDS_EULERP_IMPL_ARG_CTOR_INIT_SELF(),
                  //
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(u),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uGrad),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalar, uScalar_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarGrad, uScalarGrad_v),
                  // out
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uFL),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uFR),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uGradFL),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uGradFR),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarFL, uScalarFL_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarFR, uScalarFR_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarGradFL, uScalarGradFL_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarGradFR, uScalarGradFR_v)
            {
            }
        };

        static void Rec2nd(Rec2nd_Arg &arg);

        struct Flux2nd_Arg
        {
            Evaluator &self;
            Evaluator::t_deviceView<B> this_v; //! must keep this alive
            EvaluatorDeviceView<B> self_view;

            TUDof::t_deviceView<B> uFL;
            TUDof::t_deviceView<B> uFR;
            TUGrad::t_deviceView<B> uGradFL;
            TUGrad::t_deviceView<B> uGradFR;
            deviceViewVector<TUScalar::t_deviceView<B>, B> uScalarFL_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalarFL;
            deviceViewVector<TUScalar::t_deviceView<B>, B> uScalarFR_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalarFR;
            deviceViewVector<TUScalarGrad::t_deviceView<B>, B> uScalarGradFL_v;
            vector_DeviceView<B, TUScalarGrad::t_deviceView<B>> uScalarGradFL;
            deviceViewVector<TUScalarGrad::t_deviceView<B>, B> uScalarGradFR_v;
            vector_DeviceView<B, TUScalarGrad::t_deviceView<B>> uScalarGradFR;

            TUScalar::t_deviceView<B> muFF;
            deviceViewVector<TUScalar::t_deviceView<B>, B> muCompFF_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> muCompFF;
            TUScalar::t_deviceView<B> deltaLamFaceFF;

            // out
            TUDof::t_deviceView<B> fluxFF;
            deviceViewVector<TUScalar::t_deviceView<B>, B> fluxScalarFF_v;
            vector_DeviceView<B, TUScalar::t_deviceView<B>> fluxScalarFF;
            // out added
            TUDof::t_deviceView<B> rhs;
            deviceViewVector<TUScalarGrad::t_deviceView<B>, B> rhsScalar_v;
            vector_DeviceView<B, TUScalarGrad::t_deviceView<B>> rhsScalar;

            Flux2nd_Arg(Evaluator &self_, Evaluator::Flux2nd_Arg &arg)
                : DNDS_EULERP_IMPL_ARG_CTOR_INIT_SELF(),
                  //
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uFL),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uFR),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uGradFL),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(uGradFR),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarFL, uScalarFL_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarFR, uScalarFR_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarGradFL, uScalarGradFL_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(uScalarGradFR, uScalarGradFR_v),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_SSPARR(muFF),
                  DNDS_EULERP_IMPL_ARG_CTOR_INIT_VECSSPARR(muCompFF, muCompFF_v),
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