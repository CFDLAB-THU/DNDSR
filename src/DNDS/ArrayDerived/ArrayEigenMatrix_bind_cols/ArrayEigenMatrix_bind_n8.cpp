#include "../ArrayEigenMatrix_bind.hpp"

namespace DNDS
{
    template void pybind11_callBindArrayEigenMatrixs_rowsizes<8>(py::module_ &m); // N x 8
}