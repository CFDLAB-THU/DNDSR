#pragma once

#include "IndexMapping.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace DNDS
{
    auto pybind11_GlobalOffsetsMapping_declare(py::module_ m)
    {
        return py::class_<GlobalOffsetsMapping, std::shared_ptr<GlobalOffsetsMapping>>(m, "GlobalOffsetsMapping");
    }

    auto pybind11_GlobalOffsetsMapping_get_class(py::module_ m)
    {
        return py::class_<GlobalOffsetsMapping, std::shared_ptr<GlobalOffsetsMapping>>(m.attr("pybind11_GlobalOffsetsMapping_declare"));
    }

    void pybind11_GlobalOffsetsMapping_define(py::module_ m)
    {
        auto Py_GlobalOffsetsMapping = pybind11_GlobalOffsetsMapping_declare(m);

        Py_GlobalOffsetsMapping
            .def(py::init<>())
            .def("globalSize", &GlobalOffsetsMapping::globalSize)
            .def("setMPIAlignBcast", &GlobalOffsetsMapping::setMPIAlignBcast)
            .def(
                "RLengths",
                [](GlobalOffsetsMapping &self)
                {
                    auto &vec = self.RLengths();
                    return py::memoryview::from_buffer<index>(
                        vec.data(),
                        {vec.size()},
                        {sizeof(index)},
                        true);
                },
                py::keep_alive<0, 1>())
            .def(
                "ROffsets",
                [](GlobalOffsetsMapping &self)
                {
                    auto &vec = self.ROffsets();
                    return py::memoryview::from_buffer<index>(
                        vec.data(),
                        {vec.size()},
                        {sizeof(index)},
                        true);
                },
                py::keep_alive<0, 1>())
            .def("search", [](GlobalOffsetsMapping &self, index globalQuery)
                 { return self.search(globalQuery); })
            .def("__call__", [](GlobalOffsetsMapping &self, MPI_int rank, index val)
                 { return self.operator()(rank, val); });
    }
}