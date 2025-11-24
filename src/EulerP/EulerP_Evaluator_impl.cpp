#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "EulerP/EulerP.hpp"
#include "EulerP/EulerP_Physics.hpp"
#include "Geom/BoundaryCondition.hpp"
#include "EulerP_Evaluator_impl.hpp"
#include "EulerP_Evaluator_impl_common.hxx"

namespace DNDS::EulerP
{
    static constexpr DeviceBackend B = DeviceBackend::Host;

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

        /*********************** */
        // bc handling
        for (index iBnd = 0; iBnd < mesh.NumBnd(); iBnd++)
        {
            RecGradient_GGRec_Kernel_BndVal(self_view, arg.portable, iBnd, nVars);
        }

        /*********************** */
        // rec
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < mesh.NumCell(); iCell++)
        {
            RecGradient_GGRec_Kernel_GG(self_view, arg.portable, iCell, nVars, nVarsScalar);
        }
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

        /*********************** */
        // limit

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < mesh.NumCell(); iCell++)
        {
            RecGradient_BarthLimiter_Kernel_FlowPart(self_view, arg.portable, iCell, nVars, nVarsScalar);
        }

        // for scalars

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < mesh.NumCell(); iCell++)
        {
            RecGradient_BarthLimiter_Kernel_ScalarPart(self_view, arg.portable, iCell, nVars, nVarsScalar);
        }
    }

    template <>
    void Evaluator_impl<B>::Cons2PrimMu(Cons2PrimMu_Arg &arg)
    {
        using namespace Geom;

        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(u)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalar)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalarGrad)
        //
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uGradPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalarPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalarGradPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(p)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(T)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(a)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(gamma)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(mu)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(muComp)

        int nVarsScalar = uScalar.size();
        int nVars = nVarsFlow + nVarsScalar;

        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iPt = 0; iPt < u.Size(); iPt++)
        {
            Cons2PrimMu_Kernel(self_view, arg.portable, iPt, nVars, nVarsScalar);
        }
    }

    template <>
    void Evaluator_impl<B>::Cons2Prim(Cons2Prim_Arg &arg)
    {
        using namespace Geom;

        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(u)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalar)
        //
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalarPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(p)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(T)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(a)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(gamma)

        int nVarsScalar = uScalar.size();
        int nVars = nVarsFlow + nVarsScalar;

        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iPt = 0; iPt < u.Size(); iPt++)
        {
            Cons2Prim_Kernel(self_view, arg.portable, iPt, nVars, nVarsScalar);
        }
    }

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

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iFace = 0; iFace < mesh.NumFaceProc(); iFace++)
        {
            EstEigenDt_GetFaceLam_Kernel(self_view, arg.portable, iFace, nVars, nVarsScalar);
        }
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
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < mesh.NumCell(); iCell++)
        {
            EstEigenDt_FaceLam2CellDt_Kernel(self_view, arg.portable, iCell, nVars, nVarsScalar);
        }
    }

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

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iFace = 0; iFace < mesh.NumFaceProc(); iFace++)
        {
            RecFace2nd_Kernel(self_view, arg.portable, iFace, nVars, nVarsScalar);
        }
    }

    template <>
    void Evaluator_impl<B>::Flux2nd(Flux2nd_Arg &arg)
    {
        using namespace Geom;
        DNDS_EULERP_IMPL_ARG_GET_REF(self)
        DNDS_EULERP_IMPL_ARG_GET_REF(self_view)

        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(u)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uGrad)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalar)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalarGrad)

        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uGradPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalarPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalarGradPrim)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(p)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(T)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(a)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(gamma)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(mu)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(muComp)

        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uFL)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uFR)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uGradFF)

        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalarFL)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalarFR)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(uScalarGradFF)

        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(pFL)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(pFR)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(deltaLamFaceFF)

        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(fluxFF)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(fluxScalarFF)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(rhs)
        DNDS_EULERP_IMPL_ARG_GET_REF_PORTABLE(rhsScalar)

        auto &mesh = self_view.fv.mesh;
        auto &fv = self_view.fv;
        auto &bcHandler = self_view.bcHandler;
        auto &phy = self_view.physics;

        int nVarsScalar = uScalarFL.size();
        int nVars = nVarsScalar + nVarsFlow;

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iFace = 0; iFace < mesh.NumFaceProc(); iFace++)
        {
            Flux2nd_Kernel_FluxFace(self_view, arg.portable, iFace, nVars, nVarsScalar);
        }

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < mesh.NumCell(); iCell++)
        {
            Flux2nd_Kernel_Face2Cell(self_view, arg.portable, iCell, nVars, nVarsScalar);
        }
    }
}