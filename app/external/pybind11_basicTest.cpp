#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "DNDS/MPI.hpp"

namespace py = pybind11;

using namespace pybind11::literals; 

namespace DNDS::Python
{
    PYBIND11_MODULE(pybind11_basicTest, m)
    {
        m.def("MPI_Init", [](const std::vector<std::string> &pArgv)
              {
            std::vector<const char*> argStarts;
            for(auto & v: pArgv)
                argStarts.push_back(v.c_str()); 
            int argn = argStarts.size();
            auto argv = argStarts.data();
            auto ret = MPI_Init(&argn, const_cast<char***>(&argv) );
            //!Warning: assuming mpi won't touch anything
            return std::make_tuple(ret,pArgv); });
        m.def("MPI_Finalize", []()
              { return MPI_Finalize(); });

        py::class_<DNDS::MPIInfo>(m, "MPIInfo")
            .def(py::init<>())
            .def_property_readonly("comm", [](const MPIInfo &mpi)
                                   {
#ifndef _WIN32
                                       return intptr_t(static_cast<void *>(mpi.comm));
#else
                    return intptr_t(mpi.comm);
#endif
                                   })
            .def_property_readonly("size", [](const MPIInfo &mpi)
                                   { return mpi.size; })
            .def_property_readonly("rank", [](const MPIInfo &mpi)
                                   { return mpi.rank; })
            .def("setWorld", &MPIInfo::setWorld)
            .def("__repr__", [](const MPIInfo &mpi)
                 { return "<DNDS::MPIInfo: " + std::to_string(intptr_t(&mpi)) + ">\n" +
#ifndef _WIN32
                          " comm: " + std::to_string(intptr_t(static_cast<void *>(mpi.comm))) + "\n" +
#else
                         " comm: " + std::to_string(intptr_t((mpi.comm))) + "\n" +
#endif
                         " size: " + std::to_string(mpi.size) + "\n" +
                         " rank: " + std::to_string(mpi.rank) + "\n"; });
    }
}
