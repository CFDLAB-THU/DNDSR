/** @file EulerP_Evaluator_impl.cpp
 *  @brief Host-backend explicit specializations of Evaluator_impl kernel methods.
 *
 *  Provides the DeviceBackend::Host specialization for each Evaluator_impl static method.
 *  Each method contains an OpenMP-parallelized loop that calls the corresponding
 *  device-callable kernel function from EulerP_Evaluator_impl_common.hxx.
 *
 *  For the CUDA backend specializations, see the separate CUDA compilation unit.
 */
#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "EulerP/EulerP.hpp"
#include "EulerP_Evaluator_impl.hpp"
#include "EulerP_Evaluator_impl_common.hxx"

namespace DNDS::EulerP
{
    static constexpr DeviceBackend B = DeviceBackend::Host;

    /**
     * @brief Host specialization: Green-Gauss gradient reconstruction.
     *
     * First loop (serial): generates boundary ghost values for all boundary faces.
     * Second loop (OpenMP parallel): computes the Green-Gauss cell gradient for all owned cells.
     */
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

        /*********************** */
        // bc handling
        for (index iBnd = 0; iBnd < mesh.NumBnd(); iBnd++)
        {
            RecGradient_GGRec_Kernel_BndVal(self_view, arg.portable, iBnd, mesh.NumBnd(), nVars, nVarsScalar);
        }

        /*********************** */
        // rec
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < mesh.NumCell(); iCell++)
        {
            RecGradient_GGRec_Kernel_GG(self_view, arg.portable, iCell, mesh.NumCell(), nVars, nVarsScalar);
        }
    }

    /**
     * @brief Host specialization: Barth-Jespersen gradient limiter.
     *
     * First loop (OpenMP parallel): applies the flow-variable limiter to all owned cells.
     * Second loop (OpenMP parallel, conditional): applies the scalar-variable limiter if
     * there are transported scalars.
     */
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
            RecGradient_BarthLimiter_Kernel_FlowPart(self_view, arg.portable, iCell, mesh.NumCell(), nVars, nVarsScalar);
        }

        // for scalars

        if (nVarsScalar)
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < mesh.NumCell(); iCell++)
            {
                RecGradient_BarthLimiter_Kernel_ScalarPart(self_view, arg.portable, iCell, mesh.NumCell(), nVars, nVarsScalar);
            }
    }

    /**
     * @brief Host specialization: conservative-to-primitive conversion with viscosity.
     *
     * OpenMP-parallel loop over all points in the u array, calling Cons2PrimMu_Kernel
     * for each cell.
     */
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

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iPt = 0; iPt < u.Size(); iPt++)
        {
            Cons2PrimMu_Kernel(self_view, arg.portable, iPt, u.Size(), nVars, nVarsScalar);
        }
    }

    /**
     * @brief Host specialization: conservative-to-primitive conversion (no gradients/viscosity).
     *
     * OpenMP-parallel loop over all points in the u array, calling Cons2Prim_Kernel
     * for each cell.
     */
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

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iPt = 0; iPt < u.Size(); iPt++)
        {
            Cons2Prim_Kernel(self_view, arg.portable, iPt, u.Size(), nVars, nVarsScalar);
        }
    }

    /**
     * @brief Host specialization: per-face eigenvalue estimation.
     *
     * OpenMP-parallel loop over all processor-local faces, calling
     * EstEigenDt_GetFaceLam_Kernel for each face.
     */
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
            EstEigenDt_GetFaceLam_Kernel(self_view, arg.portable, iFace, mesh.NumFaceProc(), nVars, nVarsScalar);
        }
    }

    /**
     * @brief Host specialization: face eigenvalue accumulation to cell time steps.
     *
     * OpenMP-parallel loop over all owned cells, calling
     * EstEigenDt_FaceLam2CellDt_Kernel for each cell.
     */
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
            EstEigenDt_FaceLam2CellDt_Kernel(self_view, arg.portable, iCell, mesh.NumCell(), nVars, nVarsScalar);
        }
    }

    /**
     * @brief Host specialization: 2nd-order face value reconstruction.
     *
     * OpenMP-parallel loop over all processor-local faces, calling
     * RecFace2nd_Kernel for each face.
     */
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
            RecFace2nd_Kernel(self_view, arg.portable, iFace, mesh.NumFaceProc(), nVars, nVarsScalar);
        }
    }

    /**
     * @brief Host specialization: 2nd-order Roe flux evaluation and face-to-cell RHS scatter.
     *
     * First loop (OpenMP parallel): computes per-face numerical flux via Flux2nd_Kernel_FluxFace.
     * Second loop (OpenMP parallel): scatters face fluxes to cell RHS via Flux2nd_Kernel_Face2Cell.
     */
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

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iFace = 0; iFace < mesh.NumFaceProc(); iFace++)
        {
            Flux2nd_Kernel_FluxFace(self_view, arg.portable, iFace, mesh.NumFaceProc(), nVars, nVarsScalar);
        }

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < mesh.NumCell(); iCell++)
        {
            Flux2nd_Kernel_Face2Cell(self_view, arg.portable, iCell, mesh.NumCell(), nVars, nVarsScalar);
        }
    }
}