#pragma once

#include "DNDS/Defines_bind.hpp"
#include "EulerP_Evaluator.hpp"

namespace DNDS::EulerP
{
    inline void pybind11_Evaluator_define(py::module_ &m)
    {
        using T = Evaluator;
        auto T_ = py_class_ssp<T>(
            m, "Evaluator");

        T_.def(py::init<T::t_fv, T::t_bcHandler, T::t_physics>());

        T_
            .def_readwrite("fv", &T::fv)
            .def_readwrite("bcHandler", &T::bcHandler)
            .def_readwrite("physics", &T::physics);

        T_
            .def("PrintDataVTKHDF", &T::PrintDataVTKHDF,
                 py::arg("fname"), py::arg("series_name"),
                 py::arg("arrCellCentScalar"),
                 py::arg("arrCellCentScalar_names"),
                 py::arg("arrCellCentVec"),
                 py::arg("arrCellCentVec_names"),
                 py::arg("arrNodeScalar"),
                 py::arg("arrNodeScalar_names"),
                 py::arg("arrNodeVec"),
                 py::arg("arrNodeVec_names"),
                 py::arg("t"));
    }

    inline void pybind11_Evaluator_bind(py::module_ &m)
    {
        pybind11_Evaluator_define(m);
    }
}