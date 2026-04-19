/** @file EulerP_Evaluator_impl.hpp
 *  @brief Backend-specific implementation layer for EulerP Evaluator kernel dispatch.
 *
 *  Defines Evaluator_impl<B>, a template struct parameterized by DeviceBackend (Host or CUDA).
 *  For each Evaluator kernel, an inner argument struct wraps the host-side shared_ptr-based
 *  Evaluator::*_Arg into device views suitable for kernel execution. The inner Portable
 *  sub-struct is trivially copyable and can be passed to CUDA kernels.
 *
 *  Also defines helper macros for constructing device views from Evaluator argument members.
 */
#pragma once

#include "EulerP_Evaluator.hpp"
#include <memory>
#include <type_traits>

namespace DNDS::EulerP
{
/** @name Argument construction macros
 *  @brief Helper macros used inside Evaluator_impl inner arg struct constructors to
 *         create device views from host-side Evaluator argument members.
 *  @{
 */
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

/** @} */ // end of argument construction macros

    /**
     * @brief Backend-specific implementation of EulerP Evaluator kernels.
     *
     * Each kernel (RecGradient, Cons2PrimMu, EstEigenDt, RecFace2nd, Flux2nd) is wrapped
     * in an inner Arg struct that converts host-side shared_ptr arrays into device views,
     * and a static method that dispatches the actual computation. The inner Portable struct
     * within each Arg is trivially copyable for CUDA kernel launch.
     *
     * @tparam B The DeviceBackend (Host or CUDA). Explicit specializations live in
     *           EulerP_Evaluator_impl.cpp (Host) and the CUDA compilation unit.
     */
    template <DeviceBackend B>
    struct Evaluator_impl
    {
        /** @brief Unique pointer to a device view vector of scalar array views. */
        using t_Scalar_deviceViewVector_sup = std::unique_ptr<deviceViewVector<TUScalar::t_deviceView<B>, B>>;
        /** @brief Unique pointer to a device view vector of scalar-gradient array views. */
        using t_ScalarGrad_deviceViewVector_sup = std::unique_ptr<deviceViewVector<TUScalarGrad::t_deviceView<B>, B>>;
        /**
         * @brief Device-side argument struct for gradient reconstruction kernels.
         *
         * Wraps Evaluator::RecGradient_Arg by converting shared_ptr arrays into device views.
         * The Portable sub-struct is trivially copyable for CUDA kernel parameters.
         */
        struct RecGradient_Arg
        {
            Evaluator &self;
            Evaluator::t_deviceView<B> this_v; //! must keep this alive
            EvaluatorDeviceView<B> self_view;

            /** @brief Trivially-copyable payload holding device views of all kernel data. */
            struct Portable
            {
                // buffer
                TUDof::t_deviceView<B> faceBCBuffer;                                      /**< @brief Face BC ghost DOF buffer (device view). */
                vector_DeviceView<B, TUScalar::t_deviceView<B>> faceBCScalarBuffer;       /**< @brief Face BC ghost scalar buffers (device view). */

                TUDof::t_deviceView<B> u;                                                 /**< @brief Conservative state (device view). */
                TUGrad::t_deviceView<B> uGrad;                                            /**< @brief Gradient of conservative state (device view). */
                vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalar;                  /**< @brief Transported scalar fields (device view). */
                vector_DeviceView<B, TUScalarGrad::t_deviceView<B>> uScalarGrad;          /**< @brief Gradients of transported scalars (device view). */
            } portable;
            static_assert(std::is_trivially_copyable_v<Portable>);

            t_Scalar_deviceViewVector_sup faceBCScalarBuffer_v;  /**< @brief Owning storage for face BC scalar buffer device views. */
            // out
            t_Scalar_deviceViewVector_sup uScalar_v;               /**< @brief Owning storage for scalar field device views. */
            t_ScalarGrad_deviceViewVector_sup uScalarGrad_v;       /**< @brief Owning storage for scalar gradient device views. */

            /**
             * @brief Constructs device views from the host-side RecGradient_Arg.
             * @param self_ Reference to the Evaluator.
             * @param arg Host-side argument struct providing shared_ptr arrays.
             */
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
        /**
         * @brief Green-Gauss gradient reconstruction: boundary ghost values + cell gradient computation.
         * @param arg Device-side argument struct with all required views.
         */
        static void RecGradient_GGRec(RecGradient_Arg &arg);

        /**
         * @brief Barth-Jespersen gradient limiter applied to reconstructed gradients.
         * @param arg Device-side argument struct with all required views.
         */
        static void RecGradient_BarthLimiter(RecGradient_Arg &arg);

        /**
         * @brief Device-side argument struct for conservative-to-primitive + viscosity kernel.
         *
         * Wraps Evaluator::Cons2PrimMu_Arg by converting all shared_ptr arrays into device views.
         */
        struct Cons2PrimMu_Arg
        {
            Evaluator &self;
            Evaluator::t_deviceView<B> this_v; //! must keep this alive
            EvaluatorDeviceView<B> self_view;

            /** @brief Trivially-copyable payload for Cons2PrimMu kernel data. */
            struct Portable
            {
                TUDof::t_deviceView<B> u;                                                 /**< @brief Conservative state (device view). */
                TUGrad::t_deviceView<B> uGrad;                                            /**< @brief Gradient of conservative state (device view). */
                vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalar;                  /**< @brief Transported scalars (device view). */
                vector_DeviceView<B, TUScalarGrad::t_deviceView<B>> uScalarGrad;          /**< @brief Scalar gradients (device view). */
                TUDof::t_deviceView<B> uPrim;                                             /**< @brief Primitive state output (device view). */
                TUGrad::t_deviceView<B> uGradPrim;                                        /**< @brief Primitive gradient output (device view). */
                vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalarPrim;              /**< @brief Primitive scalars output (device view). */
                vector_DeviceView<B, TUScalarGrad::t_deviceView<B>> uScalarGradPrim;      /**< @brief Primitive scalar gradients output (device view). */
                TUScalar::t_deviceView<B> p;                                              /**< @brief Pressure output (device view). */
                TUScalar::t_deviceView<B> T;                                              /**< @brief Temperature output (device view). */
                TUScalar::t_deviceView<B> a;                                              /**< @brief Speed of sound output (device view). */
                TUScalar::t_deviceView<B> gamma;                                          /**< @brief Gamma output (device view). */
                TUScalar::t_deviceView<B> mu;                                             /**< @brief Total viscosity output (device view). */
                vector_DeviceView<B, TUScalar::t_deviceView<B>> muComp;                   /**< @brief Component viscosities output (device view). */
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

        /**
         * @brief Executes conservative-to-primitive conversion with viscosity computation.
         * @param arg Device-side argument struct with all required views.
         */
        static void Cons2PrimMu(Cons2PrimMu_Arg &arg);

        /**
         * @brief Device-side argument struct for conservative-to-primitive conversion (no gradients/viscosity).
         */
        struct Cons2Prim_Arg
        {
            Evaluator &self;
            Evaluator::t_deviceView<B> this_v; //! must keep this alive
            EvaluatorDeviceView<B> self_view;

            struct Portable
            {
                TUDof::t_deviceView<B> u;
                vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalar;
                TUDof::t_deviceView<B> uPrim;
                vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalarPrim;

                TUScalar::t_deviceView<B> p;
                TUScalar::t_deviceView<B> T;
                TUScalar::t_deviceView<B> a;
                TUScalar::t_deviceView<B> gamma;
            } portable;
            static_assert(std::is_trivially_copyable_v<Portable>);

            t_Scalar_deviceViewVector_sup uScalar_v;
            // out
            t_Scalar_deviceViewVector_sup uScalarPrim_v;

            Cons2Prim_Arg(Evaluator &self_, Evaluator::Cons2Prim_Arg &arg)
                : DNDS_EULERP_IMPL_ARG_CTOR_INIT_SELF()
            {
                //
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(u)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(uScalar, uScalar_v)
                // out
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(uPrim)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(uScalarPrim, uScalarPrim_v)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(p)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(T)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(a)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(gamma)
            }
        };

        /**
         * @brief Executes conservative-to-primitive conversion (no gradients/viscosity).
         * @param arg Device-side argument struct with all required views.
         */
        static void Cons2Prim(Cons2Prim_Arg &arg);

        /**
         * @brief Device-side argument struct for eigenvalue estimation and time-step computation.
         */
        struct EstEigenDt_Arg
        {
            Evaluator &self;
            Evaluator::t_deviceView<B> this_v; //! must keep this alive
            EvaluatorDeviceView<B> self_view;

            struct Portable
            {
                TUDof::t_deviceView<B> u;
                TUScalar::t_deviceView<B> muCell;
                TUScalar::t_deviceView<B> aCell;
                TUScalarGrad::t_deviceView<B> faceLamEst;
                TUScalar::t_deviceView<B> faceLamVisEst;
                TUScalar::t_deviceView<B> deltaLamFace;
                TUScalar::t_deviceView<B> deltaLamCell;
                TUScalar::t_deviceView<B> dt;
            } portable;
            static_assert(std::is_trivially_copyable_v<Portable>);

            EstEigenDt_Arg(Evaluator &self_, Evaluator::EstEigenDt_Arg &arg)
                : DNDS_EULERP_IMPL_ARG_CTOR_INIT_SELF()
            {
                //
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(u)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(muCell)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(aCell)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(faceLamEst)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(faceLamVisEst)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(deltaLamFace)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(deltaLamCell)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(dt)
            }
        };

        /**
         * @brief First pass: computes per-face eigenvalue estimates from cell states.
         * @param arg Device-side argument struct with all required views.
         */
        static void EstEigenDt_GetFaceLam(EstEigenDt_Arg &arg);

        /**
         * @brief Second pass: accumulates face eigenvalues to cells and computes local dt.
         * @param arg Device-side argument struct with all required views.
         */
        static void EstEigenDt_FaceLam2CellDt(EstEigenDt_Arg &arg);

        /**
         * @brief Device-side argument struct for 2nd-order face value reconstruction.
         */
        struct RecFace2nd_Arg
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

                // out
                TUDof::t_deviceView<B> uFL;
                TUDof::t_deviceView<B> uFR;
                TUGrad::t_deviceView<B> uGradFF;
                vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalarFL;
                vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalarFR;
                vector_DeviceView<B, TUScalarGrad::t_deviceView<B>> uScalarGradFF;
            } portable;
            static_assert(std::is_trivially_copyable_v<Portable>);

            t_Scalar_deviceViewVector_sup uScalar_v;
            t_ScalarGrad_deviceViewVector_sup uScalarGrad_v;
            t_Scalar_deviceViewVector_sup uScalarFL_v;
            t_Scalar_deviceViewVector_sup uScalarFR_v;
            t_ScalarGrad_deviceViewVector_sup uScalarGradFF_v;

            RecFace2nd_Arg(Evaluator &self_, Evaluator::RecFace2nd_Arg &arg)
                : DNDS_EULERP_IMPL_ARG_CTOR_INIT_SELF()
            {
                //
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(u)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(uGrad)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(uScalar, uScalar_v)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(uScalarGrad, uScalarGrad_v)
                // out
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(uFL)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(uFR)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(uGradFF)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(uScalarFL, uScalarFL_v)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(uScalarFR, uScalarFR_v)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(uScalarGradFF, uScalarGradFF_v)
            }
        };

        /**
         * @brief Executes 2nd-order face value reconstruction from cell-centered data.
         * @param arg Device-side argument struct with all required views.
         */
        static void RecFace2nd(RecFace2nd_Arg &arg);

        /**
         * @brief Device-side argument struct for 2nd-order flux evaluation and RHS accumulation.
         *
         * The largest impl arg struct, wrapping all state, thermodynamic, face-reconstructed,
         * and output arrays as device views for kernel execution.
         */
        struct Flux2nd_Arg
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
                TUScalar::t_deviceView<B> deltaLamCell;

                TUDof::t_deviceView<B> uFL;
                TUDof::t_deviceView<B> uFR;
                TUGrad::t_deviceView<B> uGradFF;

                vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalarFL;
                vector_DeviceView<B, TUScalar::t_deviceView<B>> uScalarFR;
                vector_DeviceView<B, TUScalarGrad::t_deviceView<B>> uScalarGradFF;

                TUScalar::t_deviceView<B> pFL;
                TUScalar::t_deviceView<B> pFR;

                vector_DeviceView<B, TUScalar::t_deviceView<B>> fluxScalarFF;

                // out
                TUDof::t_deviceView<B> fluxFF;
                // out added
                TUDof::t_deviceView<B> rhs;
                vector_DeviceView<B, TUScalar::t_deviceView<B>> rhsScalar;
            } portable;
            static_assert(std::is_trivially_copyable_v<Portable>);

            t_Scalar_deviceViewVector_sup uScalar_v;
            t_ScalarGrad_deviceViewVector_sup uScalarGrad_v;

            t_Scalar_deviceViewVector_sup uScalarPrim_v;
            t_ScalarGrad_deviceViewVector_sup uScalarGradPrim_v;
            t_Scalar_deviceViewVector_sup muComp_v;

            t_Scalar_deviceViewVector_sup uScalarFL_v;
            t_Scalar_deviceViewVector_sup uScalarFR_v;
            t_ScalarGrad_deviceViewVector_sup uScalarGradFF_v;

            t_Scalar_deviceViewVector_sup fluxScalarFF_v;
            t_Scalar_deviceViewVector_sup rhsScalar_v;

            Flux2nd_Arg(Evaluator &self_, Evaluator::Flux2nd_Arg &arg)
                : DNDS_EULERP_IMPL_ARG_CTOR_INIT_SELF()
            {
                //
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(u)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(uGrad)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(uScalar, uScalar_v)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(uScalarGrad, uScalarGrad_v)

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
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(deltaLamCell)
                //
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(uFL)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(uFR)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(uGradFF)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(uScalarFL, uScalarFL_v)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(uScalarFR, uScalarFR_v)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(uScalarGradFF, uScalarGradFF_v)

                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(pFL)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(pFR)
                // out
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(fluxFF)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(fluxScalarFF, fluxScalarFF_v)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_SSPARR(rhs)
                DNDS_EULERP_IMPL_ARG_CTOR_PORTABLE_COPY_VECSSPARR(rhsScalar, rhsScalar_v)
            }
        };

        /**
         * @brief Evaluates 2nd-order Roe flux per face and scatters to cell RHS.
         * @param arg Device-side argument struct with all required views.
         */
        static void Flux2nd(Flux2nd_Arg &arg);
    };
}