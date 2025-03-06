
#include "MPI_bind.hpp"
#include "Array_bind.hpp"

PYBIND11_MODULE(dnds_pybind11, m)
{
    DNDS::pybind11_MPIInfo(m);
    DNDS::MPI::pybind11_Init_thread(m);
    DNDS::MPI::pybind11_MPI_Operations(m);
    DNDS::Debug::pybind11_Debug(m);

    DNDS::pybind11_array_define<DNDS::real, 3>(m);
    DNDS::pybind11_array_define<DNDS::real, DNDS::NonUniformSize>(m);
}
