
#include "MPI_bind.hpp"
#include "IndexMapping_bind.hpp"
#include "Array_bind.hpp"
#include "ArrayDerived/ArrayAdjacency_bind.hpp"

PYBIND11_MODULE(dnds_pybind11, m)
{
    DNDS::pybind11_bind_MPI_All(m);
    
    DNDS::pybind11_bind_IndexMapping_All(m);

    DNDS::pybind11_bind_Array_All(m);

    DNDS::pybind11_bind_ArrayAdjacency_All(m);
}
