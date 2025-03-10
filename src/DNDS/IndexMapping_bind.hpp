#pragma once

#include "IndexMapping.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace DNDS
{
    inline auto pybind11_GlobalOffsetsMapping_declare(py::module_ m)
    {
        return py::class_<GlobalOffsetsMapping, std::shared_ptr<GlobalOffsetsMapping>>(m, "GlobalOffsetsMapping");
    }

    inline auto pybind11_GlobalOffsetsMapping_get_class(py::module_ m)
    {
        return py::class_<GlobalOffsetsMapping, std::shared_ptr<GlobalOffsetsMapping>>(m.attr("GlobalOffsetsMapping"));
    }

    inline void pybind11_GlobalOffsetsMapping_define(py::module_ m)
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
                    return py_vector_as_memory_view(self.RLengths(), true);
                },
                py::keep_alive<0, 1>())
            .def(
                "ROffsets",
                [](GlobalOffsetsMapping &self)
                {
                    return py_vector_as_memory_view(self.ROffsets(), true);
                },
                py::keep_alive<0, 1>())
            .def("search", [](GlobalOffsetsMapping &self, index globalQuery)
                 { return self.search(globalQuery); })
            .def("__call__", [](GlobalOffsetsMapping &self, MPI_int rank, index val)
                 { return self.operator()(rank, val); });
    }

    inline auto pybind11_OffsetAscendIndexMapping_declare(py::module_ m)
    {
        return py::class_<OffsetAscendIndexMapping, std::shared_ptr<OffsetAscendIndexMapping>>(m, "OffsetAscendIndexMapping");
    }

    inline auto pybind11_OffsetAscendIndexMapping_get_class(py::module_ m)
    {
        return py::class_<OffsetAscendIndexMapping, std::shared_ptr<OffsetAscendIndexMapping>>(m.attr("OffsetAscendIndexMapping"));
    }

    inline void pybind11_OffsetAscendIndexMapping_define(py::module_ m)
    {
        auto Py_OffsetAscendIndexMapping = pybind11_OffsetAscendIndexMapping_declare(m);

        Py_OffsetAscendIndexMapping
            .def(
                py::init(
                    [](index nmainOffset, index nmainSize,
                       std::vector<index> pullingIndexGlobal,
                       const GlobalOffsetsMapping &LGlobalMapping, const MPIInfo &mpi)
                    { return std::make_shared<OffsetAscendIndexMapping>(nmainOffset, nmainSize, pullingIndexGlobal, LGlobalMapping, mpi); }),
                py::arg("nmainOffset"), py::arg("nmainSize"),
                py::arg("pullingIndexGlobal"),
                py::arg("LGlobalMapping"), py::arg("mpi"))
            .def(
                py::init([](index nmainOffset, index nmainSize,
                            std::vector<index> pushingIndexesLocal, // which stores local index
                            std::vector<index> pushingStarts,
                            const GlobalOffsetsMapping &LGlobalMapping,
                            const MPIInfo &mpi)
                         { return std::make_shared<OffsetAscendIndexMapping>(
                               nmainOffset, nmainSize, pushingIndexesLocal, pushingStarts, LGlobalMapping, mpi); }),
                py::arg("nmainOffset"), py::arg("nmainSize"),
                py::arg("pushingIndexLocal"), py::arg("pushingStarts"),
                py::arg("LGlobalMapping"), py::arg("mpi"))
            .def("ghostAt", &OffsetAscendIndexMapping::ghostAt)
            .def_property_readonly(
                "ghostIndex",
                [](OffsetAscendIndexMapping &self)
                {
                    return py_vector_as_memory_view(self.ghostIndex, true);
                },
                py::keep_alive<0, 1>())
            .def_property_readonly(
                "ghostSizes",
                [](OffsetAscendIndexMapping &self)
                {
                    return py_vector_as_memory_view(self.ghostSizes, true);
                },
                py::keep_alive<0, 1>())
            .def_property_readonly(
                "ghostStart",
                [](OffsetAscendIndexMapping &self)
                {
                    return py_vector_as_memory_view(self.ghostStart, true);
                },
                py::keep_alive<0, 1>())
            .def_property_readonly(
                "pullingRequestLocal",
                [](OffsetAscendIndexMapping &self)
                {
                    return py_vector_as_memory_view(self.pullingRequestLocal, true);
                },
                py::keep_alive<0, 1>())
            .def_property_readonly(
                "pushingIndexGlobal",
                [](OffsetAscendIndexMapping &self)
                {
                    return py_vector_as_memory_view(self.pushingIndexGlobal, true);
                },
                py::keep_alive<0, 1>())
            .def_property_readonly(
                "pushIndexSizes",
                [](OffsetAscendIndexMapping &self)
                {
                    return py_vector_as_memory_view(self.pushIndexSizes, true);
                },
                py::keep_alive<0, 1>())
            .def_property_readonly(
                "pushIndexStarts",
                [](OffsetAscendIndexMapping &self)
                {
                    return py_vector_as_memory_view(self.pushIndexStarts, true);
                },
                py::keep_alive<0, 1>())
            .def("search", [&](OffsetAscendIndexMapping &self, index globalQuery)
                 { return self.search(globalQuery); }, py::arg("globalQuery"))
            .def("search_indexAppend", [&](OffsetAscendIndexMapping &self, index globalQuery)
                 { return self.search_indexAppend(globalQuery); }, py::arg("globalQuery"))
            .def("search_indexRank", [&](OffsetAscendIndexMapping &self, index globalQuery)
                 { return self.search_indexRank(globalQuery); }, py::arg("globalQuery"))
            .def("__call__", [&](OffsetAscendIndexMapping &self, MPI_int rank, index val)
                 { return self.operator()(rank, val); }, py::arg("rank"), py::arg("val"));
        ;
    }

    inline void pybind11_bind_IndexMapping_All(py::module_ m)
    {
        pybind11_GlobalOffsetsMapping_define(m);
        pybind11_OffsetAscendIndexMapping_define(m);
    }
}