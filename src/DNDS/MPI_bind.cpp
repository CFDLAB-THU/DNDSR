#include "MPI.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace DNDS
{
    void pybind11_MPIInfo(py::module_ &m)
    {
        py::class_<MPIInfo>(m, "MPIInfo")
            .def(py::init<>())
            .def("setWorld", &MPIInfo::setWorld)
            .def_readonly("rank", &MPIInfo::rank)
            .def_readonly("size", &MPIInfo::size)
            .def("comm", [](const MPIInfo &mpi)
                 { return size_t(mpi.comm); })
            .def("equals", &MPIInfo::operator==);
    }
}

namespace DNDS::MPI
{
    void pybind11_Init_thread(py::module_ &m)
    {
        m.def("Init_thread",
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
        m.def("Finalize",
              []()
              {
                  return MPI_Finalize();
              });
    }

}

PYBIND11_MODULE(dnds_pybind11, m)
{
    DNDS::pybind11_MPIInfo(m);
    DNDS::MPI::pybind11_Init_thread(m);
}
