
#include "MPI_bind.hpp"
#include "IndexMapping_bind.hpp"
#include "Array_bind.hpp"

PYBIND11_MODULE(dnds_pybind11, m)
{
    DNDS::pybind11_MPIInfo(m);
    DNDS::MPI::pybind11_Init_thread(m);
    DNDS::MPI::pybind11_MPI_Operations(m);
    DNDS::Debug::pybind11_Debug(m);

    DNDS::pybind11_GlobalOffsetsMapping_define(m);

    DNDS::pybind11_callBindArrays_rowsizes<DNDS::real>(m);
    DNDS::pybind11_array_define<DNDS::real, DNDS::DynamicSize>(m);
    DNDS::pybind11_array_define<DNDS::real, DNDS::NonUniformSize>(m);

    DNDS::pybind11_callBindArrays_rowsizes<DNDS::index>(m);
    DNDS::pybind11_array_define<DNDS::index, DNDS::DynamicSize>(m);
    DNDS::pybind11_array_define<DNDS::index, DNDS::NonUniformSize>(m);

    DNDS::pybind11_callBindParArrays_rowsizes<DNDS::real>(m);
    DNDS::pybind11_pararray_define<DNDS::real, DNDS::DynamicSize>(m);
    DNDS::pybind11_pararray_define<DNDS::real, DNDS::NonUniformSize>(m);

    DNDS::pybind11_callBindParArrays_rowsizes<DNDS::index>(m);
    DNDS::pybind11_pararray_define<DNDS::index, DNDS::DynamicSize>(m);
    DNDS::pybind11_pararray_define<DNDS::index, DNDS::NonUniformSize>(m);
}
