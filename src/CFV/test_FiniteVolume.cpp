#include "test_FiniteVolume.hpp"

namespace DNDS::CFV
{
    static const DeviceBackend B = DeviceBackend::Host;

    template <>
    void finiteVolumeCellOpTest_run<B>(
        FiniteVolume::t_deviceView<B> &fv,
        tUDof<DynamicSize>::t_deviceView<B> &u,
        tUGrad<DynamicSize, 3>::t_deviceView<B> &u_grad,
        const t_jsonconfig &settings)
    {
#pragma omp parallel for
        for (index iCell = 0; iCell < fv.mesh.NumCell(); iCell++)
            finiteVolumeCellOpTest(fv, u, u_grad, iCell);
    }
}