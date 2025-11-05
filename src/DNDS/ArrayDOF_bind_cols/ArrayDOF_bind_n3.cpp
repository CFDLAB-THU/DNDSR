#include "../ArrayDOF_bind.hpp"

namespace DNDS
{
    template void pybind11_callBindArrayDOF_rowsizes<3>(py::module_ &m);
}