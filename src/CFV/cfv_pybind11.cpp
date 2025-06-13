
#include "VRDefines_bind.hpp"
#include "VariationalReconstruction_bind.hpp"

PYBIND11_MODULE(cfv_pybind11, m)
{
    auto m_dnds = py::module_::import("DNDS");
    auto m_geom = py::module_::import("Geom");
    using namespace DNDS::CFV;

    pybind11_VRDefines_define(m, m_dnds);

    pybind11_VariationalReconstruction_define<2>(m);
    pybind11_VariationalReconstruction_define<3>(m);
}
