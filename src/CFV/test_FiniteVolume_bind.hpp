#pragma once
#include "VRDefines.hpp"
#include "VRDefines_bind.hpp"
#include "test_FiniteVolume.hpp"
#include <pybind11_json/pybind11_json.hpp>

namespace DNDS::CFV
{
    template <DeviceBackend B>
    void pybind11_test_FiniteVolume_define(py::module_ &m)
    {
        m.def((std::string("finiteVolumeCellOpTest_main_") + device_backend_name(B)).c_str(),
              [](FiniteVolume &fv,
                 tUDof<DynamicSize> &u,
                 tUGrad<DynamicSize, 3> &u_grad,
                 py::object settings)
              {
                  nlohmann::json settings_json = settings;

                  finiteVolumeCellOpTest_main<B>(fv, u, u_grad, t_jsonconfig(settings_json));
              },
              py::arg("fv"), py::arg("u"), py::arg("u_grad"), py::arg("settings") = py::none());
    }

}