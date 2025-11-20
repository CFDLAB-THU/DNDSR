#pragma once

#include "EulerP_Physics.hpp"
#include "DNDS/Defines_bind.hpp"
#include <pybind11_json/pybind11_json.hpp>

namespace DNDS::EulerP
{
    inline void pybind11_Physics_bind(py::module_ &m)
    {
        auto Physics_ = py_class_ssp<Physics>(
            m, "Physics");

        Physics_.def(py::init());

        Physics_
            .def("to_dict",
                 [](Physics &self)
                 {
                     return py::dict(nlohmann::json(self));
                 })
            .def_static("from_dict",
                        [](const py::dict &d)
                        {
                            return Physics(nlohmann::json(d));
                        });
    }
}