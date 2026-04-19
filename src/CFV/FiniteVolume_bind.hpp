#pragma  once

#include "DNDS/Defines_bind.hpp"
#include "FiniteVolume.hpp"
#include "VRDefines_bind.hpp"

namespace DNDS::CFV
{
    using tPy_FiniteVolume = py_class_ssp<FiniteVolume>;

    void pybind11_FiniteVolume_define(py::module_ &m);
}