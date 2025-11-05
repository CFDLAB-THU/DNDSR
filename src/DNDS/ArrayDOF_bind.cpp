#include "ArrayDOF_bind.hpp"

namespace DNDS
{
    void pybind11_bind_ArrayDOF_All(py::module_ &m)
    {
        pybind11_callBindArrayDOF_rowsizes<1>(m);
        pybind11_callBindArrayDOF_rowsizes<2>(m);
        pybind11_callBindArrayDOF_rowsizes<3>(m);
        pybind11_callBindArrayDOF_rowsizes<4>(m);
        pybind11_callBindArrayDOF_rowsizes<5>(m);
        pybind11_callBindArrayDOF_rowsizes<6>(m);
        pybind11_callBindArrayDOF_rowsizes<7>(m);
        pybind11_callBindArrayDOF_rowsizes<8>(m);
        pybind11_callBindArrayDOF_rowsizes<DynamicSize>(m);
        pybind11_callBindArrayDOF_rowsizes<NonUniformSize>(m);
    }
}