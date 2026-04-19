/**
 * @file EulerP_BC_bind.hpp
 * @brief Pybind11 bindings for the EulerP boundary condition types and handlers.
 *
 * Exposes the following Python API under the EulerP module:
 * - @c BCType: Enum with all boundary condition types (Wall, Far, Sym, etc.)
 * - @c BCInput: JSON-serializable BC input specification with name, type, value properties
 *   and to_dict/from_dict methods
 * - @c BC: Single boundary condition with id, type, values properties
 * - @c BCHandler: BC manager constructed from (bc_inputs, name2id) with id2bc lookup
 */
#pragma once

#include "DNDS/Defines_bind.hpp"
#include "EulerP_BC.hpp"

#include <pybind11_json/pybind11_json.hpp>
#include <pybind11/stl.h>

namespace DNDS::EulerP
{
    /**
     * @brief Registers the BCType enum as a Python enum class.
     *
     * Exposes all BCType values: Wall, WallInvis, WallIsothermal, Far, Sym,
     * In, InPsTs, Out, OutP, Special, Unknown.
     *
     * @param m Pybind11 module to register the enum into.
     */
    inline void pybind11_BCType_define(py::module_ &m)
    {
#define DNDS_PY_ENUM_CLASS_BCType_ADD(v) value(#v, BCType::v)

        py::enum_<BCType>(m, "BCType")
            .DNDS_PY_ENUM_CLASS_BCType_ADD(Wall)
            .DNDS_PY_ENUM_CLASS_BCType_ADD(WallInvis)
            .DNDS_PY_ENUM_CLASS_BCType_ADD(WallIsothermal)
            .DNDS_PY_ENUM_CLASS_BCType_ADD(Far)
            .DNDS_PY_ENUM_CLASS_BCType_ADD(Sym)
            .DNDS_PY_ENUM_CLASS_BCType_ADD(In)
            .DNDS_PY_ENUM_CLASS_BCType_ADD(InPsTs)
            .DNDS_PY_ENUM_CLASS_BCType_ADD(Out)
            .DNDS_PY_ENUM_CLASS_BCType_ADD(OutP)
            .DNDS_PY_ENUM_CLASS_BCType_ADD(Special)
            .DNDS_PY_ENUM_CLASS_BCType_ADD(Unknown);
#undef DNDS_PY_ENUM_CLASS_BCType_ADD
    }

    /**
     * @brief Registers pybind11 bindings for the BCInput struct.
     *
     * Exposes BCInput with:
     * - Default constructor
     * - @c to_dict() / @c from_dict() for JSON round-trip
     * - @c name, @c type, @c value as read/write properties
     *
     * @param m Pybind11 module to register bindings into.
     */
    inline void pybind11_BCInput_define(py::module_ &m)
    {
        using T = BCInput;
        auto T_ = py_class_ssp<T>(
            m, "BCInput");

        T_.def(py::init());

        T_
            .def("to_dict",
                 [](T &self)
                 {
                     return py::dict(nlohmann::json(self));
                 })
            .def_static("from_dict",
                        [](const py::dict &d)
                        {
                            return T(nlohmann::json(d));
                        });

        T_
            .def_readwrite("name", &T::name)
            .def_readwrite("type", &T::type)
            .def_readwrite("value", &T::value);
    }

    /**
     * @brief Registers pybind11 bindings for the BC class.
     *
     * Exposes BC with:
     * - Default constructor
     * - @c id (read/write property via getId/setId)
     * - @c type (read/write property via getType/setType)
     * - @c values (read/write property via getValues/setValues)
     *
     * @param m Pybind11 module to register bindings into.
     */
    inline void pybind11_BC_define(py::module_ &m)
    {
        using T = BC;
        auto T_ = py_class_ssp<T>(
            m, "BC");

        T_
            .def(py::init());
        T_
            .def_property("id", &T::getId, &T::setId)
            .def_property("type", &T::getType, &T::setType)
            .def_property("values", &T::getValues, &T::setValues);
    }

    /**
     * @brief Registers pybind11 bindings for the BCHandler class.
     *
     * Exposes BCHandler with:
     * - Constructor taking (bc_inputs: list[BCInput], name2id: AutoAppendName2ID)
     * - @c id2bc(id) method for BC lookup by zone ID
     *
     * @param m Pybind11 module to register bindings into.
     */
    inline void pybind11_BCHandler_define(py::module_ &m)
    {
        using T = BCHandler;
        auto T_ = py_class_ssp<T>(
            m, "BCHandler");

        T_.def(py::init([](const std::vector<BCInput> &bc_inputs, Geom::AutoAppendName2ID &name2id)
                        { return BCHandler(bc_inputs, name2id); }));

        T_
            .def("id2bc", &T::id2bc);
    }

    /**
     * @brief Top-level binding function for all EulerP boundary condition Python API.
     *
     * Calls pybind11_BCType_define, pybind11_BCInput_define, pybind11_BC_define,
     * and pybind11_BCHandler_define to register the complete BC Python interface.
     *
     * @param m Pybind11 module to register bindings into.
     */
    inline void pybind11_BC_bind(py::module_ &m)
    {
        pybind11_BCType_define(m);
        pybind11_BCInput_define(m);
        pybind11_BC_define(m);
        pybind11_BCHandler_define(m);
    }
}