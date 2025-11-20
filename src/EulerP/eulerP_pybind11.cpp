#include "DNDS/Defines_bind.hpp"
#include "EulerP_Physics_bind.hpp"
#include "EulerP_BC_bind.hpp"
#include "EulerP_Evaluator_bind.hpp"

PYBIND11_MODULE(eulerP_pybind11, m)
{
    auto m_placeholder_submodule = m.def_submodule("placeholder_submodule");

    //! hard coded dependency here
    //! better solution?
    py::module_::import("DNDSR.DNDS");
    py::module_::import("DNDSR.Geom");
    py::module_::import("DNDSR.CFV");

    using namespace DNDS::EulerP;
    pybind11_Physics_bind(m);

    pybind11_BC_bind(m);

    pybind11_Evaluator_bind(m);
}
