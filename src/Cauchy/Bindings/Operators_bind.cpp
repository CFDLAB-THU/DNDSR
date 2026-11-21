#include "Cauchy/Bindings/Operators_bind.hpp"

namespace DNDS::Cauchy::Bindings
{
    void BindFiniteVolumeOperators(pybind11::module_ &m)
    {
        // Registration order matters: default arguments reference prior types.
        BindOperatorState(m);
        BindOperatorLinear(m);
        BindOperatorNonlinear(m);
        BindOperatorFiniteVolume(m);
        BindUpdatePolicies(m);
    }
}
