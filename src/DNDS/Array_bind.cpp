#include "Array_bind.hpp"

namespace DNDS
{
    void pybind11_bind_Array_All(py::module_ m)
    {
        // primitive serial Array's:
        pybind11_callBindArrays_rowsizes<real>(m);
        pybind11_Array_define<real, DynamicSize>(m);
        pybind11_Array_define<real, NonUniformSize>(m);

        pybind11_callBindArrays_rowsizes<index>(m);
        pybind11_Array_define<index, DynamicSize>(m);
        pybind11_Array_define<index, NonUniformSize>(m);

        // primitive ParArray's
        pybind11_callBindParArrays_rowsizes<real>(m);
        pybind11_ParArray_define<real, DynamicSize>(m);
        pybind11_ParArray_define<real, NonUniformSize>(m);

        pybind11_callBindParArrays_rowsizes<index>(m);
        pybind11_ParArray_define<index, DynamicSize>(m);
        pybind11_ParArray_define<index, NonUniformSize>(m);

        // primitive ArrayTransformer's
        pybind11_callBindArrayTransformers_rowsizes<real>(m);
        pybind11_ArrayTransformer_define<ParArray<real, DynamicSize>>(m);
        pybind11_ArrayTransformer_define<ParArray<real, NonUniformSize>>(m);

        pybind11_callBindArrayTransformers_rowsizes<index>(m);
        pybind11_ArrayTransformer_define<ParArray<index, DynamicSize>>(m);
        pybind11_ArrayTransformer_define<ParArray<index, NonUniformSize>>(m);
    }
}