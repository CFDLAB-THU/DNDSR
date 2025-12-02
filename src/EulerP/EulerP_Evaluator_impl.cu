#define DNDS_NDEBUG_DEVICE
#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "EulerP/EulerP.hpp"
#include "EulerP_Evaluator_impl.hpp"
#include "EulerP_Evaluator_impl_common.hxx"
#include "DNDS/CUDA_Utils.hpp"
#include "cuda_runtime.h"

#define DNDS_EULERP_SERIALIZE_CUDA_EXECUTION

namespace DNDS::EulerP
{
    static constexpr DeviceBackend B = DeviceBackend::CUDA;

#define DNDS_EULERP_DEFINE_CUDA_SIMPLE_POINTWISE_KERNEL(FNAME, ARGTYPE) \
    DNDS_GLOBAL static void CUDA_Simple_PointWise_Call_##FNAME(         \
        EvaluatorDeviceView<B> *self_view,                              \
        typename ARGTYPE arg, int nVars, int nVarsScalar,               \
        index lo, index up)                                             \
    {                                                                   \
        index tid_g = DNDS_CUDA_1D_TID_GLOBAL_INDEX;                    \
        FNAME(*self_view, arg, lo + tid_g, up, nVars, nVarsScalar);     \
    }

    DNDS_EULERP_DEFINE_CUDA_SIMPLE_POINTWISE_KERNEL(RecGradient_GGRec_Kernel_BndVal, Evaluator_impl<B>::RecGradient_Arg::Portable)
    DNDS_EULERP_DEFINE_CUDA_SIMPLE_POINTWISE_KERNEL(RecGradient_GGRec_Kernel_GG, Evaluator_impl<B>::RecGradient_Arg::Portable)
    DNDS_EULERP_DEFINE_CUDA_SIMPLE_POINTWISE_KERNEL(RecGradient_BarthLimiter_Kernel_FlowPart, Evaluator_impl<B>::RecGradient_Arg::Portable)
    DNDS_EULERP_DEFINE_CUDA_SIMPLE_POINTWISE_KERNEL(RecGradient_BarthLimiter_Kernel_ScalarPart, Evaluator_impl<B>::RecGradient_Arg::Portable)

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

        DNDS_check_throw(faceBCBuffer.father.data());
        DNDS_check_throw(faceBCBuffer.father.Size() >= mesh.NumFace());

        DNDS_CUDA_DEVICE_VIEW_COPY_OBJ(self_view)
        int threadsPerBlock0 = arg.self.get_config()["threadsPerBlock"];
        /*********************** */
        // bc handling
        {
            auto [blocksPerGrid, threadsPerBlock] =
                CUDA::calckernelSizeSimple(mesh.NumBnd(), threadsPerBlock0);
            CUDA_Simple_PointWise_Call_RecGradient_GGRec_Kernel_BndVal<<<blocksPerGrid, threadsPerBlock>>>(
                self_view_device_copy.get(), arg.portable, nVars, nVarsScalar, 0, mesh.NumBnd());
        }

        /*********************** */
        // rec
        {
            auto [blocksPerGrid, threadsPerBlock] =
                CUDA::calckernelSizeSimple(mesh.NumCell(), threadsPerBlock0);
            CUDA_Simple_PointWise_Call_RecGradient_GGRec_Kernel_GG<<<blocksPerGrid, threadsPerBlock>>>(
                self_view_device_copy.get(), arg.portable, nVars, nVarsScalar, 0, mesh.NumCell());
        }
        if (arg.self.get_config()["serializeCUDAExecution"])
            cudaDeviceSynchronize();
    }

    template <>
    void Evaluator_impl<B>::RecGradient_BarthLimiter(RecGradient_Arg &arg)
    {
        using namespace Geom;

        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(u)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalar)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalarGrad)

        int nVarsScalar = uScalar.size();
        int nVars = nVarsFlow + nVarsScalar;

        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

        DNDS_CUDA_DEVICE_VIEW_COPY_OBJ(self_view)
        int threadsPerBlock0 = arg.self.get_config()["threadsPerBlock"];
        /*********************** */
        // limit

        {
            auto [blocksPerGrid, threadsPerBlock] =
                CUDA::calckernelSizeSimple(mesh.NumCell(), threadsPerBlock0);
            CUDA_Simple_PointWise_Call_RecGradient_BarthLimiter_Kernel_FlowPart<<<blocksPerGrid, threadsPerBlock>>>(
                self_view_device_copy.get(), arg.portable, nVars, nVarsScalar, 0, mesh.NumCell());
        }

        // for scalars

        {
            auto [blocksPerGrid, threadsPerBlock] =
                CUDA::calckernelSizeSimple(mesh.NumCell(), threadsPerBlock0);
            CUDA_Simple_PointWise_Call_RecGradient_BarthLimiter_Kernel_ScalarPart<<<blocksPerGrid, threadsPerBlock>>>(
                self_view_device_copy.get(), arg.portable, nVars, nVarsScalar, 0, mesh.NumCell());
        }
        if (arg.self.get_config()["serializeCUDAExecution"])
            cudaDeviceSynchronize();
    }

    DNDS_EULERP_DEFINE_CUDA_SIMPLE_POINTWISE_KERNEL(Cons2PrimMu_Kernel, Evaluator_impl<B>::Cons2PrimMu_Arg::Portable)

    template <>
    void Evaluator_impl<B>::Cons2PrimMu(Cons2PrimMu_Arg &arg)
    {
        using namespace Geom;

        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(u)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalar)

        int nVarsScalar = uScalar.size();
        int nVars = nVarsFlow + nVarsScalar;

        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

        DNDS_CUDA_DEVICE_VIEW_COPY_OBJ(self_view)
        int threadsPerBlock0 = arg.self.get_config()["threadsPerBlock"];
        {
            auto [blocksPerGrid, threadsPerBlock] =
                CUDA::calckernelSizeSimple(u.Size(), threadsPerBlock0);
            CUDA_Simple_PointWise_Call_Cons2PrimMu_Kernel<<<blocksPerGrid, threadsPerBlock>>>(
                self_view_device_copy.get(), arg.portable, nVars, nVarsScalar, 0, u.Size());
        }
        if (arg.self.get_config()["serializeCUDAExecution"])
            cudaDeviceSynchronize();
    }

    DNDS_EULERP_DEFINE_CUDA_SIMPLE_POINTWISE_KERNEL(Cons2Prim_Kernel, Evaluator_impl<B>::Cons2Prim_Arg::Portable)

    template <>
    void Evaluator_impl<B>::Cons2Prim(Cons2Prim_Arg &arg)
    {
        using namespace Geom;

        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(u)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalar)

        int nVarsScalar = uScalar.size();
        int nVars = nVarsFlow + nVarsScalar;

        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

        DNDS_CUDA_DEVICE_VIEW_COPY_OBJ(self_view)
        int threadsPerBlock0 = arg.self.get_config()["threadsPerBlock"];
        {
            auto [blocksPerGrid, threadsPerBlock] =
                CUDA::calckernelSizeSimple(u.Size(), threadsPerBlock0);
            CUDA_Simple_PointWise_Call_Cons2Prim_Kernel<<<blocksPerGrid, threadsPerBlock>>>(
                self_view_device_copy.get(), arg.portable, nVars, nVarsScalar, 0, u.Size());
        }
        if (arg.self.get_config()["serializeCUDAExecution"])
            cudaDeviceSynchronize();
    }

    DNDS_EULERP_DEFINE_CUDA_SIMPLE_POINTWISE_KERNEL(EstEigenDt_GetFaceLam_Kernel, Evaluator_impl<B>::EstEigenDt_Arg::Portable)
    DNDS_EULERP_DEFINE_CUDA_SIMPLE_POINTWISE_KERNEL(EstEigenDt_FaceLam2CellDt_Kernel, Evaluator_impl<B>::EstEigenDt_Arg::Portable)

    template <>
    void Evaluator_impl<B>::EstEigenDt_GetFaceLam(EstEigenDt_Arg &arg)
    {
        using namespace Geom;

        DNDS_EULERP_IMPL_ARG_GET_REF(self)
        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)

        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(u)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(muCell)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(aCell)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(faceLamEst)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(faceLamVisEst)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(deltaLamFace)

        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

        int nVarsScalar = 0; // no uScalar here
        int nVars = nVarsFlow + nVarsScalar;
        DNDS_CUDA_DEVICE_VIEW_COPY_OBJ(self_view)
        int threadsPerBlock0 = arg.self.get_config()["threadsPerBlock"];
        {
            auto [blocksPerGrid, threadsPerBlock] =
                CUDA::calckernelSizeSimple(mesh.NumFaceProc(), threadsPerBlock0);
            CUDA_Simple_PointWise_Call_EstEigenDt_GetFaceLam_Kernel<<<blocksPerGrid, threadsPerBlock>>>(
                self_view_device_copy.get(), arg.portable, nVars, nVarsScalar, 0, mesh.NumFaceProc());
        }
        if (arg.self.get_config()["serializeCUDAExecution"])
            cudaDeviceSynchronize();
    }

    template <>
    void Evaluator_impl<B>::EstEigenDt_FaceLam2CellDt(EstEigenDt_Arg &arg)
    {

        DNDS_EULERP_IMPL_ARG_GET_REF(self)
        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)

        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

        int nVarsScalar = 0; // no uScalar here
        int nVars = nVarsFlow + nVarsScalar;

        DNDS_CUDA_DEVICE_VIEW_COPY_OBJ(self_view)
        int threadsPerBlock0 = arg.self.get_config()["threadsPerBlock"];
        {
            auto [blocksPerGrid, threadsPerBlock] =
                CUDA::calckernelSizeSimple(mesh.NumCell(), threadsPerBlock0);
            CUDA_Simple_PointWise_Call_EstEigenDt_FaceLam2CellDt_Kernel<<<blocksPerGrid, threadsPerBlock>>>(
                self_view_device_copy.get(), arg.portable, nVars, nVarsScalar, 0, mesh.NumCell());
        }
        if (arg.self.get_config()["serializeCUDAExecution"])
            cudaDeviceSynchronize();
    }

    DNDS_EULERP_DEFINE_CUDA_SIMPLE_POINTWISE_KERNEL(RecFace2nd_Kernel, Evaluator_impl<B>::RecFace2nd_Arg::Portable)

    template <>
    void Evaluator_impl<B>::RecFace2nd(RecFace2nd_Arg &arg)
    {
        using namespace Geom;
        DNDS_EULERP_IMPL_ARG_GET_REF(self)
        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)

        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalar)

        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

        int nVarsScalar = uScalar.size();
        int nVars = nVarsScalar + nVarsFlow;

        DNDS_CUDA_DEVICE_VIEW_COPY_OBJ(self_view)
        int threadsPerBlock0 = arg.self.get_config()["threadsPerBlock"];
        {
            auto [blocksPerGrid, threadsPerBlock] =
                CUDA::calckernelSizeSimple(mesh.NumFaceProc(), threadsPerBlock0);
            CUDA_Simple_PointWise_Call_RecFace2nd_Kernel<<<blocksPerGrid, threadsPerBlock>>>(
                self_view_device_copy.get(), arg.portable, nVars, nVarsScalar, 0, mesh.NumFaceProc());
        }
        if (arg.self.get_config()["serializeCUDAExecution"])
            cudaDeviceSynchronize();
    }

    DNDS_EULERP_DEFINE_CUDA_SIMPLE_POINTWISE_KERNEL(Flux2nd_Kernel_FluxFace, Evaluator_impl<B>::Flux2nd_Arg::Portable)
    DNDS_EULERP_DEFINE_CUDA_SIMPLE_POINTWISE_KERNEL(Flux2nd_Kernel_Face2Cell, Evaluator_impl<B>::Flux2nd_Arg::Portable)

    template <>
    void Evaluator_impl<B>::Flux2nd(Flux2nd_Arg &arg)
    {
        using namespace Geom;
        DNDS_EULERP_IMPL_ARG_GET_REF(self)
        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)

        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalarFL)

        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

        int nVarsScalar = uScalarFL.size();
        int nVars = nVarsScalar + nVarsFlow;
        DNDS_CUDA_DEVICE_VIEW_COPY_OBJ(self_view)
        int threadsPerBlock0 = arg.self.get_config()["threadsPerBlock"];
        {
            auto [blocksPerGrid, threadsPerBlock] =
                CUDA::calckernelSizeSimple(mesh.NumFaceProc(), threadsPerBlock0);
            CUDA_Simple_PointWise_Call_Flux2nd_Kernel_FluxFace<<<blocksPerGrid, threadsPerBlock>>>(
                self_view_device_copy.get(), arg.portable, nVars, nVarsScalar, 0, mesh.NumFaceProc());
        }

        {
            auto [blocksPerGrid, threadsPerBlock] =
                CUDA::calckernelSizeSimple(mesh.NumCell(), threadsPerBlock0);
            CUDA_Simple_PointWise_Call_Flux2nd_Kernel_Face2Cell<<<blocksPerGrid, threadsPerBlock>>>(
                self_view_device_copy.get(), arg.portable, nVars, nVarsScalar, 0, mesh.NumCell());
        }
        if (arg.self.get_config()["serializeCUDAExecution"])
            cudaDeviceSynchronize();
    }
}