#pragma once

#include "DNDS/Defines_bind.hpp"
#include "DNDS/Array_bind.hpp"
#include "Mesh.hpp"
#include <pybind11_json/pybind11_json.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

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
    def(#foo, &UnstructuredMesh::foo)

#define DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(m_name) \
    def_readonly(#m_name, &UnstructuredMesh::m_name, py::return_value_policy::reference_internal)

        auto UnstructuredMesh_ = tPy_UnstructuredMesh(m, "UnstructuredMesh");
        UnstructuredMesh_
            .def(py::init<>([](const MPIInfo &mpi, int n_dim)
                            { return std::make_shared<UnstructuredMesh>(mpi, n_dim); }))
            // basic
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(coords)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(cell2node)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(bnd2node)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(bnd2cell)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(cell2cell)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(cellElemInfo)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(bndElemInfo)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(cell2cellOrig)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(node2nodeOrig)
            // interpolated
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(cell2face)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(face2cell)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(face2node)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER(faceElemInfo);

        UnstructuredMesh_
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(RecoverNode2CellAndNode2Bnd)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(RecoverCell2CellAndBnd2Cell)
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
            // .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(ReorderLocalCells) //! it has argument now
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(getMPI)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(getDim)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumNode)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumCell)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumFace)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumBnd)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumNodeGhost)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumCellGhost)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumFaceGhost)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumBndGhost)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumNodeProc)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumCellProc)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumFaceProc)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumBndProc)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumNodeGlobal)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumCellGlobal)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumFaceGlobal)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(NumBndGlobal);
        //!!
        UnstructuredMesh_
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(BuildO2FromO1Elevation)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(BuildBisectO1FormO2)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(RecreatePeriodicNodes)
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(BuildVTKConnectivity);

        UnstructuredMesh_
            .DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC(getArrayBytes);
#undef DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_READONLY_MEMBER
#undef DNDS_GEOM_UNSTRUCTURED_MESH_PY_DEF_SIMP_FUNC

        UnstructuredMesh_
            .def("ReorderLocalCells", &UnstructuredMesh::ReorderLocalCells, py::arg("nParts") = 1)
            .def("ReadSerialize", &UnstructuredMesh::ReadSerialize, py::arg("serializer"), py::arg("name"))
            .def("WriteSerialize", &UnstructuredMesh::WriteSerialize, py::arg("serializer"), py::arg("name"));

        UnstructuredMesh_.def("SetPeriodicGeometry", &UnstructuredMesh::SetPeriodicGeometry,
                              py::arg("translation1") = Geom::tPoint{1, 0, 0},
                              py::arg("rotationCenter1") = Geom::tPoint{0, 0, 0},
                              py::arg("eulerAngles1") = Geom::tPoint{0, 0, 0},
                              py::arg("translation2") = Geom::tPoint{0, 1, 0},
                              py::arg("rotationCenter2") = Geom::tPoint{0, 0, 0},
                              py::arg("eulerAngles2") = Geom::tPoint{0, 0, 0},
                              py::arg("translation3") = Geom::tPoint{0, 0, 1},
                              py::arg("rotationCenter3") = Geom::tPoint{0, 0, 0},
                              py::arg("eulerAngles3") = Geom::tPoint{0, 0, 0});

        UnstructuredMesh_
            .def("CellFaceOther", &UnstructuredMesh::CellFaceOther,
                 py::arg("iCell"), py::arg("iFace"))
            .def("CellIsFaceBack", &UnstructuredMesh::CellIsFaceBack,
                 py::arg("iCell"), py::arg("iFace"));

        auto WallDistOptions_ = py::class_<UnstructuredMesh::WallDistOptions>(UnstructuredMesh_, "WallDistOptions");
        WallDistOptions_.def(py::init());
        WallDistOptions_
            .def_readwrite("method", &UnstructuredMesh::WallDistOptions::method)
            .def_readwrite("verbose", &UnstructuredMesh::WallDistOptions::verbose)
            .def_readwrite("wallDistExecution", &UnstructuredMesh::WallDistOptions::wallDistExecution)
            .def_readwrite("minWallDist", &UnstructuredMesh::WallDistOptions::minWallDist)
            .def_readwrite("subdivide_quad", &UnstructuredMesh::WallDistOptions::subdivide_quad);

        UnstructuredMesh_
            .def("BuildNodeWallDist", &UnstructuredMesh::BuildNodeWallDist, py::arg("fBndIsWall"), py::arg("options") = UnstructuredMesh::WallDistOptions{});

        UnstructuredMesh_
            .def("to_device", [](UnstructuredMesh &self, const std::string &backend)
                 { self.to_device(device_backend_name_to_enum(backend)); }, py::arg("backend"))
            .def("to_host", &UnstructuredMesh::to_host);
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
            .def_readwrite("mesh", &UnstructuredMeshSerialRW::mesh)
            .def_readonly("dataIsSerialOut", &UnstructuredMeshSerialRW::dataIsSerialOut)
            .def_readonly("dataIsSerialIn", &UnstructuredMeshSerialRW::dataIsSerialIn)
            .def("ReadFromCGNSSerial", py::overload_cast<const std::string &>(&UnstructuredMeshSerialRW::ReadFromCGNSSerial))
            .def("Deduplicate1to1Periodic", &UnstructuredMeshSerialRW::Deduplicate1to1Periodic)
            .def("BuildCell2Cell", &UnstructuredMeshSerialRW::BuildCell2Cell)
            .def("BuildSerialOut", &UnstructuredMeshSerialRW::BuildSerialOut)
            .def(
                "MeshPartitionCell2Cell",
                [](UnstructuredMeshSerialRW &self, py::object options_in) // use default
                {
                    auto options_full = UnstructuredMeshSerialRW::PartitionOptions();
                    auto json_options_full = nlohmann::ordered_json(options_full); // warning: using {} here makes it a list
                    if (!options_in.is_none())
                    {
                        auto json_options_in = nlohmann::json(options_in);
                        json_options_full.merge_patch(json_options_in);
                    }
                    // std::cout << json_options_full << std::endl;
                    self.MeshPartitionCell2Cell(json_options_full.template get<UnstructuredMeshSerialRW::PartitionOptions>());
                },
                py::arg("options") = py::none())
            .def("PartitionReorderToMeshCell2Cell", &UnstructuredMeshSerialRW::PartitionReorderToMeshCell2Cell);
    }
}