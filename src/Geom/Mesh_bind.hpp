#pragma once

#include "DNDS/Defines_bind.hpp"
#include "DNDS/Array_bind.hpp"
#include "Mesh.hpp"

namespace DNDS::Geom
{
    using tPy_ElemInfo = py::class_<ElemInfo>;

    inline void pybind11_ElemInfo_define(py::module_ &m)
    {
        auto ElemInfo_ = tPy_ElemInfo(m, "ElemInfo");
        ElemInfo_
            .def("getElemType", &ElemInfo::getElemType)
            .def("setElemType", &ElemInfo::setElemType)
            .def_readwrite("zone", &ElemInfo::zone);
    }

    inline void pybind11_ArrayElemInfo_define(py::module_ &m)
    {
        pybind11_Array_define<ElemInfo>(m);
        pybind11_ParArray_define<ElemInfo>(m);
        pybind11_ArrayTransformer_define<ParArray<ElemInfo>>(m);
        pybind11_ParArrayPair_define<ElemInfo>(m);
    }

    using tPy_UnstructuredMesh = py::class_<UnstructuredMesh, ssp<UnstructuredMesh>>;

    inline void pybind11_UnstructuredMesh_define(py::module_ &m)
    {
        auto UnstructuredMesh_ = tPy_UnstructuredMesh(m, "UnstructuredMesh");
        UnstructuredMesh_
            .def(py::init<>([](const MPIInfo &mpi, int n_dim)
                            { return std::make_shared<UnstructuredMesh>(mpi, n_dim); }))
            .def_readonly("coords", &UnstructuredMesh::coords,
                          py::return_value_policy::reference_internal)
            .def_readonly("cell2node", &UnstructuredMesh::cell2node,
                          py::return_value_policy::reference_internal)
            .def_readonly("bnd2node", &UnstructuredMesh::bnd2node,
                          py::return_value_policy::reference_internal)
            .def_readonly("bnd2cell", &UnstructuredMesh::bnd2cell,
                          py::return_value_policy::reference_internal)
            .def_readonly("cell2cell", &UnstructuredMesh::cell2cell)
            .def_readonly("cellElemInfo", &UnstructuredMesh::cellElemInfo,
                          py::return_value_policy::reference_internal)
            .def_readonly("bndElemInfo", &UnstructuredMesh::bndElemInfo,
                          py::return_value_policy::reference_internal);

        UnstructuredMesh_
            .def("BuildGhostPrimary", &UnstructuredMesh::BuildGhostPrimary)
            .def("AdjGlobal2LocalPrimary", &UnstructuredMesh::AdjGlobal2LocalPrimary)
            .def("InterpolateFace", &UnstructuredMesh::InterpolateFace)
            .def("AssertOnFaces", &UnstructuredMesh::AssertOnFaces);
        //!!
    }

    using tPy_UnstructuredMeshSerialRW = py::class_<UnstructuredMeshSerialRW, ssp<UnstructuredMeshSerialRW>>;

    inline void pybind11_UnstructuredMeshSerialRW_define(py::module_ &m)
    {
        auto UnstructuredMeshSerialRW_ = tPy_UnstructuredMeshSerialRW(m, "UnstructuredMeshSerialRW");
        UnstructuredMeshSerialRW_
            .def(py::init(
                     [](ssp<UnstructuredMesh> mesh, MPI_int mRank)
                     { return std::make_shared<UnstructuredMeshSerialRW>(mesh, mRank); }),
                 py::arg("mesh"), py::arg("mRank") = 0)
            .def_readonly("mesh", &UnstructuredMeshSerialRW::mesh)
            .def_readonly("dataIsSerialOut", &UnstructuredMeshSerialRW::dataIsSerialOut)
            .def_readonly("dataIsSerialIn", &UnstructuredMeshSerialRW::dataIsSerialIn)
            .def("ReadFromCGNSSerial", py::overload_cast<const std::string &>(&UnstructuredMeshSerialRW::ReadFromCGNSSerial))
            .def("Deduplicate1to1Periodic", &UnstructuredMeshSerialRW::Deduplicate1to1Periodic)
            .def("BuildCell2Cell", &UnstructuredMeshSerialRW::BuildCell2Cell)
            .def(
                "MeshPartitionCell2Cell",
                [](UnstructuredMeshSerialRW &self) // use default
                { self.MeshPartitionCell2Cell(UnstructuredMeshSerialRW::PartitionOptions()); })
            .def("PartitionReorderToMeshCell2Cell", &UnstructuredMeshSerialRW::PartitionReorderToMeshCell2Cell);
    }
}