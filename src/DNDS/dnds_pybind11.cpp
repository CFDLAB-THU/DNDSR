
#include "MPI_bind.hpp"
#include "IndexMapping_bind.hpp"
#include "Array_bind.hpp"
#include "ArrayDerived/ArrayAdjacency_bind.hpp"
#include "ArrayDerived/ArrayEigenMatrix_bind.hpp"
#include "ArrayDerived/ArrayEigenMatrixBatch_bind.hpp"
#include "ArrayDerived/ArrayEigenUniMatrixBatch_bind.hpp"
#include "ArrayDerived/ArrayEigenVector_bind.hpp"
#include "ArrayDOF_bind.hpp"

#include "Serializer_bind.hpp"

PYBIND11_MODULE(dnds_pybind11, m)
{
    DNDS::pybind11_bind_defines(m);

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
