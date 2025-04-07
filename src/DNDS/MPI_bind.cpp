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
            .def(py::init([](uintptr_t pComm)
                          { return std::make_unique<MPIInfo>(MPI_Comm(pComm)); }))
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
        m_MPI.def(
            "Init_thread",
            [](const std::vector<std::string> &pArgv)
            {
                // std::vector<const char *> argStarts;
                // for (auto &v : pArgv)
                //     argStarts.push_back(v.c_str());
                // int argn = argStarts.size();
                // auto argv = argStarts.data();
                // auto ret = Init_thread(&argn, const_cast<char ***>(&argv));
                // //! Warning: assuming mpi won't touch anything
                // return std::make_tuple(ret, pArgv);

                // ! a complete version

                int initial_argc = static_cast<int>(pArgv.size());
                int initial_argc_mine = initial_argc;

                // Create an array of pointers to C-style strings:
                char **argv_array = new char *[initial_argc + 1]; // +1 for NULL terminator
                char **argv_array_mine = argv_array;

                // Allocate memory and copy each string into a modifiable buffer
                for (int i = 0; i < initial_argc; ++i)
                {
                    const std::string &s = pArgv[i];
                    size_t len = s.length();
                    char *cstr = new char[len + 1]; // Allocate space for '\0'
                    strcpy(cstr, s.c_str());        // Copy the string
                    argv_array[i] = cstr;
                }
                argv_array[initial_argc] = nullptr; // NULL terminator

                int *pargc = &initial_argc;
                char ***pargv = &argv_array;

                auto ret = Init_thread(pargc, pargv);

                // Capture the modified arguments into output_args:
                std::vector<std::string> pArgvOut;
                for (int i = 0; i < *pargc; ++i)
                    pArgvOut.push_back(std::string(argv_array[i]));

                // Cleanup all dynamically allocated memory
                // Note: Even if MPI changes entries in the array, our pointers still point to
                //       our original allocations (assuming MPI doesn't reallocate the entire array)
                for (int i = 0; i <= initial_argc_mine; ++i)
                    delete[] argv_array_mine[i]; // Free each string buffer
                delete[] argv_array_mine;        // Free the pointer array

                return std::make_tuple(ret, pArgvOut);
            });
        m_MPI.def(
            "Finalize",
            []()
            {
                return MPI::Finalize();
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

namespace DNDS
{
    void pybind11_bind_MPI_All(py::module_ &m)
    {
        pybind11_MPIInfo(m);
        MPI::pybind11_Init_thread(m);
        MPI::pybind11_MPI_Operations(m);
        Debug::pybind11_Debug(m);
    }
}
