#include "ArrayEigenMatrix_bind.hpp"

namespace DNDS
{
    void pybind11_bind_ArrayEigenMatrix_All(py::module_ &m)
    {
        pybind11_callBindArrayEigenMatrixs_rowsizes<1>(m);   // N x 1
        pybind11_callBindArrayEigenMatrixs_rowsizes<2>(m);   // N x 2
        pybind11_callBindArrayEigenMatrixs_rowsizes<3>(m);   // N x 3
        pybind11_callBindArrayEigenMatrixs_rowsizes<4>(m);   // N x 4
        pybind11_callBindArrayEigenMatrixs_rowsizes<5>(m);   // N x 5
        pybind11_callBindArrayEigenMatrixs_rowsizes<6>(m);   // N x 6
        pybind11_callBindArrayEigenMatrixs_rowsizes<7>(m);   // N x 7
        pybind11_callBindArrayEigenMatrixs_rowsizes<8>(m);   // N x 8
        
        pybind11_ArrayEigenMatrix_define<DynamicSize, 1>(m); // D x 1
        pybind11_ArrayEigenMatrix_define<DynamicSize, 2>(m); // D x 1
        pybind11_ArrayEigenMatrix_define<DynamicSize, 3>(m); // D x 1
        pybind11_ArrayEigenMatrix_define<DynamicSize, 4>(m); // D x 1
        pybind11_ArrayEigenMatrix_define<DynamicSize, 5>(m); // D x 1
        pybind11_ArrayEigenMatrix_define<DynamicSize, 6>(m); // D x 1
        pybind11_ArrayEigenMatrix_define<DynamicSize, 7>(m); // D x 1
        pybind11_ArrayEigenMatrix_define<DynamicSize, 8>(m); // D x 1
        // ! not using NonUniformSize-d ArrayEigenMatrix for now
        // pybind11_ArrayEigenMatrix_define<NonUniformSize, NonUniformSize>(m);
    }
}