#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "EulerP/EulerP.hpp"
#include "EulerP_Evaluator_impl.hpp"
#include "EulerP_Evaluator_impl_common.hxx"
#include "DNDS/CUDA_Utils.hpp"

namespace DNDS::EulerP
{
    static constexpr DeviceBackend B = DeviceBackend::CUDA;

#define DNDS_EULERP_DEFINE_CUDA_SIMPLE_POINTWISE_KERNEL(FNAME, ARGTYPE)            \
    DNDS_GLOBAL static void CUDA_Simple_PointWise_Call_##FNAME(                    \
        EvaluatorDeviceView<B> *self_view,                                         \
        typename ARGTYPE arg, int nVars, int nVarsScalar)                          \
    {                                                                              \
        FNAME(*self_view, arg, DNDS_CUDA_1D_TID_GLOBAL_INDEX, nVars, nVarsScalar); \
    }

    DNDS_EULERP_DEFINE_CUDA_SIMPLE_POINTWISE_KERNEL(RecGradient_GGRec_Kernel_BndVal, Evaluator_impl<B>::RecGradient_Arg::Portable)
    DNDS_EULERP_DEFINE_CUDA_SIMPLE_POINTWISE_KERNEL(RecGradient_GGRec_Kernel_GG, Evaluator_impl<B>::RecGradient_Arg::Portable)

    template <>
    void Evaluator_impl<B>::RecGradient_GGRec(RecGradient_Arg &arg)
    {
        using namespace Geom;

        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)

        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(u)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalar)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalarGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(faceBCBuffer)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(faceBCScalarBuffer)

        int nVarsScalar = uScalar.size();
        int nVars = nVarsFlow + nVarsScalar;

        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

        DNDS_assert(faceBCBuffer.father.data());
        DNDS_assert(faceBCBuffer.father.Size() >= mesh.NumFace());

        DNDS_CUDA_DEVICE_VIEW_COPY_OBJ(self_view)
        /*********************** */
        // bc handling
        {
            auto [blocksPerGrid, threadsPerBlock] =
                CUDA::calckernelSizeSimple(mesh.NumBnd(), 64);
            CUDA_Simple_PointWise_Call_RecGradient_GGRec_Kernel_BndVal<<<blocksPerGrid, threadsPerBlock>>>(
                self_view_device_copy.get(), arg.portable, nVars, nVarsScalar);
        }

        /*********************** */
        // rec
        {
            auto [blocksPerGrid, threadsPerBlock] =
                CUDA::calckernelSizeSimple(mesh.NumCell(), 64);
            CUDA_Simple_PointWise_Call_RecGradient_GGRec_Kernel_GG<<<blocksPerGrid, threadsPerBlock>>>(
                self_view_device_copy.get(), arg.portable, nVars, nVarsScalar);
        }
    }
}