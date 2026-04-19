#include "CFV/VariationalReconstruction_Reconstruction.hxx"
#include <vector>
#define DNDS_NDEBUG
#include "BenchmarkFiniteVolume.hpp"

namespace DNDS::CFV
{
    static const DeviceBackend B = DeviceBackend::Host;

    template <>
    void finiteVolumeCellOpTest_run<B>(
        FiniteVolume::t_deviceView<B> &fv,
        tUDof<DynamicSize>::t_deviceView<B> &u,
        tUGrad<DynamicSize, 3>::t_deviceView<B> &u_grad,
        int nIter,
        const t_jsonconfig &settings)
    {
        int nVars = u.father.MatRowSize();
        for (int i = 0; i < nIter; i++)
#pragma omp parallel for
            for (index iCell = 0; iCell < fv.mesh.NumCell(); iCell++)
            {
                std::vector<real> b(nVars * 3);
                finiteVolumeCellOpTest(fv, u, u_grad, iCell, b.data());
            };
    }

    /*************************************************************/

    /*************************************************************/

    /*************************************************************/

    /*************************************************************/

    template <int nVarsFixed>
    void finiteVolumeCellOpTest_Fixed_entry<B, nVarsFixed>::run(
        FiniteVolume::t_deviceView<B> &fv,
        typename tUDof<nVarsFixed>::template t_deviceView<B> &u,
        typename tUGrad<nVarsFixed, 3>::template t_deviceView<B> &u_grad,
        int nIter,
        const t_jsonconfig &settings)
    {
        int nVars = u.father.MatRowSize();
        for (int i = 0; i < nIter; i++)
#pragma omp parallel for
            for (index iCell = 0; iCell < fv.mesh.NumCell(); iCell++)
            {
                finiteVolumeCellOpTest_Fixed<B, nVarsFixed>(fv, u, u_grad, iCell);
            };
    }

    template void finiteVolumeCellOpTest_Fixed_entry<B, 1>::run(
        FiniteVolume::t_deviceView<B> &fv,
        typename tUDof<1>::template t_deviceView<B> &u,
        typename tUGrad<1, 3>::template t_deviceView<B> &u_grad,
        int nIter,
        const t_jsonconfig &settings);

    template void finiteVolumeCellOpTest_Fixed_entry<B, 5>::run(
        FiniteVolume::t_deviceView<B> &fv,
        typename tUDof<5>::template t_deviceView<B> &u,
        typename tUGrad<5, 3>::template t_deviceView<B> &u_grad,
        int nIter,
        const t_jsonconfig &settings);

    /*************************************************************/

    /*************************************************************/

    /*************************************************************/

    /*************************************************************/

    template <int nVarsFixed>
    void finiteVolumeCellOpTest_SOA_ver0_entry<B, nVarsFixed>::run(
        FiniteVolume::t_deviceView<B> &fv,
        std::array<tUDof<1>::t_deviceView<B>, nVarsFixed> &u,
        std::array<tUGrad<1, 3>::t_deviceView<B>, nVarsFixed> &u_grad,
        int nIter,
        const t_jsonconfig &settings)
    {
        int nVars = nVarsFixed;
        for (int i = 0; i < nIter; i++)
#pragma omp parallel for
            for (index iCell = 0; iCell < fv.mesh.NumCell(); iCell++)
            {
                finiteVolumeCellOpTest_SOA_ver0<B, nVarsFixed>(fv, u, u_grad, iCell);
            };
    }

    template void finiteVolumeCellOpTest_SOA_ver0_entry<B, 1>::run(
        FiniteVolume::t_deviceView<B> &fv,
        std::array<tUDof<1>::t_deviceView<B>, 1> &u,
        std::array<tUGrad<1, 3>::t_deviceView<B>, 1> &u_grad,
        int nIter,
        const t_jsonconfig &settings);

    template void finiteVolumeCellOpTest_SOA_ver0_entry<B, 5>::run(
        FiniteVolume::t_deviceView<B> &fv,
        std::array<tUDof<1>::t_deviceView<B>, 5> &u,
        std::array<tUGrad<1, 3>::t_deviceView<B>, 5> &u_grad,
        int nIter,
        const t_jsonconfig &settings);
}