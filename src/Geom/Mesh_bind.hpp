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
#define DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(foo) \
    def(#foo, &UnstructuredMesh::##foo)

#define DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(m_name) \
    def_readonly(#m_name, &UnstructuredMesh::##m_name, py::return_value_policy::reference_internal)

        auto UnstructuredMesh_ = tPy_UnstructuredMesh(m, "UnstructuredMesh");
        UnstructuredMesh_
            .def(py::init<>([](const MPIInfo &mpi, int n_dim)
                            { return std::make_shared<UnstructuredMesh>(mpi, n_dim); }))
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(coords)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(cell2node)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(bnd2node)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(bnd2cell)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(cell2cell)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(cellElemInfo)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(bndElemInfo);

        UnstructuredMesh_
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(BuildGhostPrimary)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjGlobal2LocalPrimary)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjLocal2GlobalPrimary)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjGlobal2LocalPrimaryForBnd)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjLocal2GlobalPrimaryForBnd)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjGlobal2LocalFacial)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjLocal2GlobalFacial)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjGlobal2LocalC2F)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjLocal2GlobalC2F)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(BuildGhostN2CB)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjGlobal2LocalN2CB)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AdjLocal2GlobalN2CB)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AssertOnN2CB)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(InterpolateFace)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(AssertOnFaces)
            .def("ConstructBndMesh", [](UnstructuredMesh &self, ssp<UnstructuredMesh> &pbMesh)
                 { self.ConstructBndMesh(*pbMesh); }, py::arg("bndMesh"))
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(GetCell2CellFaceVLocal)
            // ObtainLocalFactFillOrdering
            // ObtainSymmetricSymbolicFactorization
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(ReorderLocalCells)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumNode)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumCell)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumFace)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumBnd)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumNodeGhost)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumCellGhost)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumFaceGhost)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumNodeProc)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumCellProc)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumFaceProc);
        //!!

#undef DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER
#undef DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC
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