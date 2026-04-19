#include "ArrayEigenMatrix_bind.hpp"

namespace DNDS
{
    void pybind11_bind_ArrayEigenMatrix_All(py::module_ &m)
    {
        // extern built:
        pybind11_callBindArrayEigenMatrixs_rowsizes<1>(m);              // N x 1
        pybind11_callBindArrayEigenMatrixs_rowsizes<2>(m);              // N x 2
        pybind11_callBindArrayEigenMatrixs_rowsizes<3>(m);              // N x 3
        pybind11_callBindArrayEigenMatrixs_rowsizes<4>(m);              // N x 4
        pybind11_callBindArrayEigenMatrixs_rowsizes<5>(m);              // N x 5
        pybind11_callBindArrayEigenMatrixs_rowsizes<6>(m);              // N x 6
        pybind11_callBindArrayEigenMatrixs_rowsizes<7>(m);              // N x 7
        pybind11_callBindArrayEigenMatrixs_rowsizes<8>(m);              // N x 8
        pybind11_callBindArrayEigenMatrixs_rowsizes<DynamicSize>(m);    // N x D
        pybind11_callBindArrayEigenMatrixs_rowsizes<NonUniformSize>(m); // N x I

        // locally built:

        pybind11_ArrayEigenMatrix_define<DynamicSize, 1>(m); // D x 1
        pybind11_ArrayEigenMatrix_define<DynamicSize, 2>(m); // D x 2
        pybind11_ArrayEigenMatrix_define<DynamicSize, 3>(m); // D x 3
        pybind11_ArrayEigenMatrix_define<DynamicSize, 4>(m); // D x 4
        pybind11_ArrayEigenMatrix_define<DynamicSize, 5>(m); // D x 5
        pybind11_ArrayEigenMatrix_define<DynamicSize, 6>(m); // D x 6
        pybind11_ArrayEigenMatrix_define<DynamicSize, 7>(m); // D x 7
        pybind11_ArrayEigenMatrix_define<DynamicSize, 8>(m); // D x 8
        pybind11_ArrayEigenMatrix_define<DynamicSize, DynamicSize>(m);
        // ! not using NonUniformSize-d ArrayEigenMatrix for now

        pybind11_ArrayEigenMatrix_define<DynamicSize, NonUniformSize>(m);    // D x I
        pybind11_ArrayEigenMatrix_define<NonUniformSize, 1>(m);              // I x 1
        pybind11_ArrayEigenMatrix_define<NonUniformSize, 2>(m);              // I x 2
        pybind11_ArrayEigenMatrix_define<NonUniformSize, 3>(m);              // I x 3
        pybind11_ArrayEigenMatrix_define<NonUniformSize, 4>(m);              // I x 4
        pybind11_ArrayEigenMatrix_define<NonUniformSize, 5>(m);              // I x 5
        pybind11_ArrayEigenMatrix_define<NonUniformSize, 6>(m);              // I x 6
        pybind11_ArrayEigenMatrix_define<NonUniformSize, 7>(m);              // I x 7
        pybind11_ArrayEigenMatrix_define<NonUniformSize, 8>(m);              // I x 8
        pybind11_ArrayEigenMatrix_define<NonUniformSize, DynamicSize>(m);    // I x D
        pybind11_ArrayEigenMatrix_define<NonUniformSize, NonUniformSize>(m); // I x I

        pybind11_ArrayEigenMatrixPair_define<DynamicSize, 1>(m);           // D x 1
        pybind11_ArrayEigenMatrixPair_define<DynamicSize, 2>(m);           // D x 2
        pybind11_ArrayEigenMatrixPair_define<DynamicSize, 3>(m);           // D x 3
        pybind11_ArrayEigenMatrixPair_define<DynamicSize, 4>(m);           // D x 4
        pybind11_ArrayEigenMatrixPair_define<DynamicSize, 5>(m);           // D x 5
        pybind11_ArrayEigenMatrixPair_define<DynamicSize, 6>(m);           // D x 6
        pybind11_ArrayEigenMatrixPair_define<DynamicSize, 7>(m);           // D x 7
        pybind11_ArrayEigenMatrixPair_define<DynamicSize, 8>(m);           // D x 8
        pybind11_ArrayEigenMatrixPair_define<DynamicSize, DynamicSize>(m); // D x D
        // ! not using NonUniformSize-d ArrayEigenMatrixPair for now

        pybind11_ArrayEigenMatrixPair_define<DynamicSize, NonUniformSize>(m);    // D x I
        pybind11_ArrayEigenMatrixPair_define<NonUniformSize, 1>(m);              // I x 1
        pybind11_ArrayEigenMatrixPair_define<NonUniformSize, 2>(m);              // I x 2
        pybind11_ArrayEigenMatrixPair_define<NonUniformSize, 3>(m);              // I x 3
        pybind11_ArrayEigenMatrixPair_define<NonUniformSize, 4>(m);              // I x 4
        pybind11_ArrayEigenMatrixPair_define<NonUniformSize, 5>(m);              // I x 5
        pybind11_ArrayEigenMatrixPair_define<NonUniformSize, 6>(m);              // I x 6
        pybind11_ArrayEigenMatrixPair_define<NonUniformSize, 7>(m);              // I x 7
        pybind11_ArrayEigenMatrixPair_define<NonUniformSize, 8>(m);              // I x 8
        pybind11_ArrayEigenMatrixPair_define<NonUniformSize, DynamicSize>(m);    // I x D
        pybind11_ArrayEigenMatrixPair_define<NonUniformSize, NonUniformSize>(m); // I x I
    }
}