/// @file dnds_pybind11.cpp
/// @brief pybind11 entry point for the `dnds_pybind11` Python module.
///
/// Dispatches to the per-header `pybind11_bind_*` helpers so every DNDS
/// component can register its own bindings in isolation. The resulting
/// shared library is loaded by `DNDSR.DNDS` on the Python side.

#include "DNDS/DeviceStorage_bind.hpp"
#include "MPI_bind.hpp"
#include "DeviceStorage_bind.hpp"
#include "IndexMapping_bind.hpp"
#include "Array_bind.hpp"
#include "ArrayDerived/ArrayAdjacency_bind.hpp"
#include "ArrayDerived/ArrayEigenMatrix_bind.hpp"
#include "ArrayDerived/ArrayEigenMatrixBatch_bind.hpp"
#include "ArrayDerived/ArrayEigenUniMatrixBatch_bind.hpp"
#include "ArrayDerived/ArrayEigenVector_bind.hpp"
#include "ArrayDOF_bind.hpp"

#include "Serializer_bind.hpp"

/// @brief pybind11 module initialiser for `dnds_pybind11`.
/// @details Calls each component's `pybind11_bind_*` function against the
/// shared module object, in a safe dependency order (Defines, DeviceStorage,
/// MPI, IndexMapping, Array, ArrayDerived, ArrayDOF, Serializer).
PYBIND11_MODULE(dnds_pybind11, m)
{
    DNDS::pybind11_bind_defines(m);

    DNDS::pybind11_bind_deviceStorage(m);

    DNDS::pybind11_bind_MPI_All(m);

    DNDS::pybind11_bind_IndexMapping_All(m);

    DNDS::pybind11_bind_Array_All(m);

    DNDS::pybind11_bind_Array_Offsets(m);

    DNDS::pybind11_bind_ArrayAdjacency_All(m);

    DNDS::pybind11_bind_ArrayEigenMatrix_All(m);

    DNDS::pybind11_bind_ArrayEigenMatrixBatch_All(m);

    DNDS::pybind11_bind_ArrayEigenUniMatrixBatch_All(m);

    DNDS::pybind11_bind_ArrayEigenVector_All(m);

    DNDS::pybind11_bind_ArrayDOF_All(m);

    DNDS::Serializer::pybind11_bind_Serializer(m);
}
