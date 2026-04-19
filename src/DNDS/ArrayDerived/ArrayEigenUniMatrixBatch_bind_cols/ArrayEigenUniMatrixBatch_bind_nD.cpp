#include "../ArrayEigenUniMatrixBatch_bind.hpp"

namespace DNDS
{
    template void pybind11_callBindArrayEigenUniMatrixBatchs_rowsizes<DynamicSize>(py::module_ &m);
}