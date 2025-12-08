
#include "VRDefines_bind.hpp"
#include "FiniteVolume_bind.hpp"
#include "VariationalReconstruction_bind.hpp"
#include "ModelEvaluator_bind.hpp"
#include "test_FiniteVolume_bind.hpp"

PYBIND11_MODULE(cfv_pybind11, m)
{
    //! hard coded dependency here
    //! better solution?
    auto m_dnds = py::module_::import("DNDSR.DNDS");
    auto m_geom = py::module_::import("DNDSR.Geom");
    using namespace DNDS::CFV;

    auto m_placeholder_submodule = m.def_submodule("placeholder_submodule");

    pybind11_VRDefines_define(m, m_dnds);

    pybind11_FiniteVolume_define(m);

    pybind11_VariationalReconstruction_define<2>(m);
    pybind11_VariationalReconstruction_define<3>(m);

    pybind11_ModelEvaluator_define(m);

    pybind11_test_FiniteVolume_define<DNDS::DeviceBackend::Host>(m);
    pybind11_test_FiniteVolume_define_Fixed<DNDS::DeviceBackend::Host, 1>(m);
    pybind11_test_FiniteVolume_define_Fixed<DNDS::DeviceBackend::Host, 5>(m);
    pybind11_test_FiniteVolume_define_SOA_ver0<DNDS::DeviceBackend::Host, 1>(m);
    pybind11_test_FiniteVolume_define_SOA_ver0<DNDS::DeviceBackend::Host, 5>(m);
#ifdef DNDS_USE_CUDA
    pybind11_test_FiniteVolume_define<DNDS::DeviceBackend::CUDA>(m);
    pybind11_test_FiniteVolume_define_Fixed<DNDS::DeviceBackend::CUDA, 1>(m);
    pybind11_test_FiniteVolume_define_Fixed<DNDS::DeviceBackend::CUDA, 5>(m);
    pybind11_test_FiniteVolume_define_SOA_ver0<DNDS::DeviceBackend::CUDA, 1>(m);
    pybind11_test_FiniteVolume_define_SOA_ver0<DNDS::DeviceBackend::CUDA, 5>(m);
#endif
}
