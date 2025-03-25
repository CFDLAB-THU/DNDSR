#include "ArrayEigenVector_bind.hpp"

namespace DNDS
{
    void pybind11_bind_ArrayEigenVector_All(py::module_ &m)
    {
        pybind11_callBindArrayEigenVectors_rowsizes(m);
        pybind11_ArrayEigenVector_define<DynamicSize>(m);
        pybind11_ArrayEigenVector_define<NonUniformSize>(m);
    }
}