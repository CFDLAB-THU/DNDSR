#pragma once

#include "DNDS/Defines_bind.hpp"
#include "DNDS/Array_bind.hpp"
#include <pybind11/functional.h>
#include <pybind11_json/pybind11_json.hpp>

#include "VariationalReconstruction.hpp"

#include "ModelEvaluator.hpp"

namespace DNDS::CFV
{
    using tPy_ModelEvaluator = py_class_ssp<ModelEvaluator>;

    void pybind11_ModelEvaluator_define(py::module_ &m);
}