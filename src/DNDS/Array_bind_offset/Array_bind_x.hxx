#pragma once
#include "../Array_bind.hpp"

namespace DNDS
{

#define pybind11_bind_Array_All_X_define(offset)                       \
    void pybind11_bind_Array_All_##offset(py::module_ m)               \
    {                                                                  \
        pybind11_callBindArrays_rowsizes<real, offset>(m);             \
                                                                       \
        pybind11_callBindArrays_rowsizes<index, offset>(m);            \
                                                                       \
        pybind11_callBindParArrays_rowsizes<real, offset>(m);          \
                                                                       \
        pybind11_callBindParArrays_rowsizes<index, offset>(m);         \
                                                                       \
        pybind11_callBindArrayTransformers_rowsizes<real, offset>(m);  \
                                                                       \
        pybind11_callBindArrayTransformers_rowsizes<index, offset>(m); \
    }

}