#include "ArrayAdjacency_bind.hpp"

namespace DNDS
{
    void pybind11_bind_ArrayAdjacency_All(py::module_ &m)
    {
        pybind11_callBindArrayAdjacencys_rowsizes(m);
        pybind11_ArrayAdjacency_define<DynamicSize>(m);
        pybind11_ArrayAdjacency_define<NonUniformSize>(m);
    }
}