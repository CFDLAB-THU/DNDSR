#include "MPI.hpp"

#include "MPI_bind.hpp"

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace DNDS
{
    void pybind11_MPIInfo(py::module_ &m)
    {
        py::class_<MPIInfo>(m, "MPIInfo")
            .def(py::init<>())
            .def("setWorld", &MPIInfo::setWorld)
            .def_readonly("rank", &MPIInfo::rank)
            .def_readonly("size", &MPIInfo::size)
            .def("comm", [](const MPIInfo &self)
                 { return size_t(self.comm); })
            .def("equals", &MPIInfo::operator==);
    }
}

namespace DNDS::MPI
{
    void pybind11_Init_thread(py::module_ &m)
    {
        auto m_MPI = m.def_submodule("MPI");
        m_MPI.def("Init_thread",
                  [](const std::vector<std::string> &pArgv)
                  {
                      std::vector<const char *> argStarts;
                      for (auto &v : pArgv)
                          argStarts.push_back(v.c_str());
                      int argn = argStarts.size();
                      auto argv = argStarts.data();
                      auto ret = Init_thread(&argn, const_cast<char ***>(&argv));
                      //! Warning: assuming mpi won't touch anything
                      return std::make_tuple(ret, pArgv);
                  });
        m_MPI.def("Finalize",
                  []()
                  {
                      return MPI_Finalize();
                  });
        m_MPI.def("GetMPIThreadLevel", &GetMPIThreadLevel);
    }

    void pybind11_MPI_Operations(py::module_ &m)
    {
        auto m_MPI = m.def_submodule("MPI");
        m_MPI.def(
            "Allreduce",
            [](py::buffer py_sendbuf, py::buffer py_recvbuf, const std::string &op, const MPIInfo &mpi)
            {
                auto send_info = py_sendbuf.request(false);
                auto recv_info = py_recvbuf.request(true);

                DNDS_assert_info(recv_info.format == send_info.format,
                                 fmt::format("send and recv buffer format incompatible: [{}], [{}]",
                                             send_info.format, recv_info.format));

                MPI_Datatype datatype = py_get_buffer_basic_mpi_datatype(send_info);
                DNDS_assert(datatype != MPI_DATATYPE_NULL);
                MPI_Op mpi_op = py_get_simple_mpi_op_by_name(op);

                auto [count_s, send_style] = py_buffer_get_contigious_size(send_info);
                auto [count_r, recv_style] = py_buffer_get_contigious_size(send_info);
                DNDS_assert_info(count_r >= count_s, "receive buffer size not enough");

                MPI_int err = Allreduce(send_info.ptr, recv_info.ptr, count_s,
                                        datatype, mpi_op, mpi.comm);
            },
            py::arg("send"), py::arg("recv"), py::arg("op"), py::arg("mpi"));
    }
}

namespace DNDS::Debug
{
    // bool IsDebugged();
    // void MPIDebugHold(const MPIInfo &mpi);

    void pybind11_Debug(py::module_ &m)
    {
        auto m_Debug = m.def_submodule("Debug");
        m_Debug.def("IsDebugged", &IsDebugged);
        m_Debug.def("MPIDebugHold", &MPIDebugHold);
    }
}

