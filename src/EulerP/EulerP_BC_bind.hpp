#pragma once

#include "DNDS/Defines_bind.hpp"
#include "EulerP_BC.hpp"

#include <pybind11_json/pybind11_json.hpp>
#include <pybind11/stl.h>

namespace DNDS::EulerP
{
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
    }

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

    inline void pybind11_BC_bind(py::module_ &m)
    {
        pybind11_BCType_define(m);
        pybind11_BCInput_define(m);
        pybind11_BC_define(m);
        pybind11_BCHandler_define(m);
    }
}