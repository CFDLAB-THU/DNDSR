#pragma once

#include <pybind11/stl.h>

#include "DNDS/Defines_bind.hpp"
#include "BoundaryCondition.hpp"

namespace DNDS::Geom
{

    using tPy_AutoAppendName2ID = py_class_ssp<AutoAppendName2ID>;

    inline void pybind11_AutoAppendName2ID_define(py::module_ &m)
    {
        auto AutoAppendName2ID_ = tPy_AutoAppendName2ID(m, "AutoAppendName2ID");
        AutoAppendName2ID_
            .def(py::init<>())
            .def_readonly("n2id_map", &AutoAppendName2ID::n2id_map)
            .def_readonly("id_cap", &AutoAppendName2ID::id_cap)
            .def("__call__", &AutoAppendName2ID::operator());
    }
}