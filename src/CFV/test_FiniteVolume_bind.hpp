#pragma once
#include "VRDefines.hpp"
#include "VRDefines_bind.hpp"
#include "test_FiniteVolume.hpp"
#include <pybind11_json/pybind11_json.hpp>
#include <string>

namespace DNDS::CFV
{
    template <DeviceBackend B>
    void pybind11_test_FiniteVolume_define(py::module_ &m)
    {
        m.def((std::string("finiteVolumeCellOpTest_main_") + device_backend_name(B)).c_str(),
              [](FiniteVolume &fv,
                 tUDof<DynamicSize> &u,
                 tUGrad<DynamicSize, 3> &u_grad,
                 int nIter,
                 py::object settings)
              {
                  nlohmann::json settings_json = settings;

                  finiteVolumeCellOpTest_main<B>(fv, u, u_grad, nIter, t_jsonconfig(settings_json));
              },
              py::arg("fv"), py::arg("u"), py::arg("u_grad"), py::arg("nIter") = 1, py::arg("settings") = py::none());
    }

    template <DeviceBackend B, int nVarsFixed>
    void pybind11_test_FiniteVolume_define_Fixed(py::module_ &m)
    {
        m.def((std::string("finiteVolumeCellOpTest_Fixed_main_") + device_backend_name(B) + "_N" + std::to_string(nVarsFixed)).c_str(),
              [](FiniteVolume &fv,
                 tUDof<nVarsFixed> &u,
                 tUGrad<nVarsFixed, 3> &u_grad,
                 int nIter,
                 py::object settings)
              {
                  nlohmann::json settings_json = settings;

                  finiteVolumeCellOpTest_Fixed_main<B, nVarsFixed>(fv, u, u_grad, nIter, t_jsonconfig(settings_json));
              },
              py::arg("fv"), py::arg("u"), py::arg("u_grad"), py::arg("nIter") = 1, py::arg("settings") = py::none());
    }

    template <DeviceBackend B, int nVarsFixed>
    void pybind11_test_FiniteVolume_define_SOA_ver0(py::module_ &m)
    {
        m.def((std::string("finiteVolumeCellOpTest_SOA_ver0_main_") + device_backend_name(B) + "_N" + std::to_string(nVarsFixed)).c_str(),
              [](FiniteVolume &fv,
                 std::array<tUDof<1>, nVarsFixed> &u,
                 std::array<tUGrad<1, 3>, nVarsFixed> &u_grad,
                 int nIter,
                 py::object settings)
              {
                  nlohmann::json settings_json = settings;

                  finiteVolumeCellOpTest_SOA_ver0_main<B, nVarsFixed>(fv, u, u_grad, nIter, t_jsonconfig(settings_json));
              },
              py::arg("fv"), py::arg("u"), py::arg("u_grad"), py::arg("nIter") = 1, py::arg("settings") = py::none());
    }
}