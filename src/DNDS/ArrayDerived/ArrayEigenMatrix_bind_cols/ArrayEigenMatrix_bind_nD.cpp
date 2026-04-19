#include "../ArrayEigenMatrix_bind.hpp"

namespace DNDS
{
    template void pybind11_callBindArrayEigenMatrixs_rowsizes<DynamicSize>(py::module_ &m); // N x 8
}