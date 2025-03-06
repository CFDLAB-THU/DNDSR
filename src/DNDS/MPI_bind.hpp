#pragma once

#include "MPI.hpp"
#include "Defines_bind.hpp"
namespace py = pybind11;

namespace DNDS
{

    inline MPI_Datatype py_get_buffer_basic_mpi_datatype(const py::buffer_info &info)
    {
        if (py_buffer_contains_T<real>(info))
            return DNDS_MPI_REAL;
        else if (py_buffer_contains_T<index>(info))
            return DNDS_MPI_INDEX;
        else if (py_buffer_contains_T<int8_t>(info))
            return MPI_INT8_T;
        else if (py_buffer_contains_T<uint8_t>(info))
            return MPI_UINT8_T;
        else if (py_buffer_contains_T<int32_t>(info))
            return MPI_INT32_T;
        else if (py_buffer_contains_T<uint32_t>(info))
            return MPI_UINT32_T;
        else if (py_buffer_contains_T<int64_t>(info))
            return MPI_INT64_T;
        else if (py_buffer_contains_T<uint64_t>(info))
            return MPI_UINT64_T;
        else if (py_buffer_contains_T<float>(info))
            return MPI_FLOAT;
        else if (py_buffer_contains_T<int16_t>(info))
            return MPI_INT16_T;
        else if (py_buffer_contains_T<uint16_t>(info))
            return MPI_UINT16_T;
        else
            DNDS_assert_info(false, "MPI datatype not found for info.format == " + info.format);

        return MPI_DATATYPE_NULL;
    }

    inline MPI_Op py_get_simple_mpi_op_by_name(const std::string &op)
    {
        MPI_Op mpi_op = MPI_OP_NULL;
        if (op == "MPI_SUM")
            mpi_op = MPI_SUM;
        else if (op == "MPI_MAX")
            mpi_op = MPI_MAX;
        else if (op == "MPI_MIN")
            mpi_op = MPI_MIN;
        else if (op == "MPI_LXOR")
            mpi_op = MPI_LXOR;
        else if (op == "MPI_BXOR")
            mpi_op = MPI_BXOR;
        else
            DNDS_assert_info(false, "MPI simple op not found: " + op);
        return mpi_op;
    }
}

namespace DNDS
{
    void pybind11_MPIInfo(py::module_ &m);
}

namespace DNDS::MPI
{
    void pybind11_Init_thread(py::module_ &m);

    void pybind11_MPI_Operations(py::module_ &m);
}

namespace DNDS::Debug
{
    void pybind11_Debug(py::module_ &m);
}
