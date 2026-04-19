/**
 * @file EulerP_Physics_bind.hpp
 * @brief Pybind11 bindings for the EulerP Physics class.
 *
 * Exposes the @c Physics class to Python with:
 * - Default constructor
 * - @c to_dict() for JSON-compatible dict serialization
 * - @c from_dict() static factory for constructing Physics from a Python dict
 */
#pragma once

#include "EulerP_Physics.hpp"
#include "DNDS/Defines_bind.hpp"
#include <pybind11_json/pybind11_json.hpp>

namespace DNDS::EulerP
{
    /**
     * @brief Registers pybind11 bindings for the Physics class.
     *
     * Exposes Physics as a shared_ptr-held class with:
     * - @c Physics() default constructor
     * - @c to_dict() → Python dict (via nlohmann_json serialization)
     * - @c Physics.from_dict(dict) → Physics (static factory via JSON deserialization)
     *
     * @param m Pybind11 module to register bindings into.
     */
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