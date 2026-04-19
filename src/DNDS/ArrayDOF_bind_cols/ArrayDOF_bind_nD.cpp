#include "../ArrayDOF_bind.hpp"

namespace DNDS
{
    template void pybind11_callBindArrayDOF_rowsizes<DynamicSize>(py::module_ &m);
}