#include "FiniteVolume_bind.hpp"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11_json/pybind11_json.hpp>

namespace DNDS::CFV
{
    void pybind11_FiniteVolume_define(py::module_ &m)
    {
        using T = FiniteVolume;
        auto FiniteVolume_ = tPy_FiniteVolume(m, "FiniteVolume");
        FiniteVolume_.def(py::init([](MPIInfo &mpi, ssp<Geom::UnstructuredMesh> mesh)
                                   { return std::make_shared<T>(mpi, mesh); }));
        FiniteVolume_
            .def(
                "GetSettings", [](T &self)
                { 
                   nlohmann::ordered_json outJson;
                   self.getSettings().WriteIntoJson(outJson);
                return py::object{outJson}; });
        FiniteVolume_
            .def(
                "ParseSettings", [](T &self, py::object settings)
                { 
                    FiniteVolumeSettings defaultSettings(self.getDim());
                    nlohmann::ordered_json defaultJson;
                    defaultSettings.WriteIntoJson(defaultJson);
                    nlohmann::json settings_json = settings;
                    defaultJson.merge_patch(settings_json);
                    self.parseSettings(defaultJson); },
                py::arg("settings"));

#define DNDS_PY_DEF_SIMP_FUNC(foo) \
    def(#foo, &T::foo)

        FiniteVolume_
            .DNDS_PY_DEF_SIMP_FUNC(SetCellAtrBasic)
            .

            DNDS_PY_DEF_SIMP_FUNC(ConstructCellVolume)
            .DNDS_PY_DEF_SIMP_FUNC(ConstructCellBary)
            .DNDS_PY_DEF_SIMP_FUNC(ConstructCellCent)
            .DNDS_PY_DEF_SIMP_FUNC(ConstructCellIntJacobiDet)
            .DNDS_PY_DEF_SIMP_FUNC(ConstructCellIntPPhysics)
            .DNDS_PY_DEF_SIMP_FUNC(ConstructCellAlignedHBox)
            . // note this is AABB
            DNDS_PY_DEF_SIMP_FUNC(ConstructCellMajorHBoxCoordInertia)
            .

            DNDS_PY_DEF_SIMP_FUNC(SetFaceAtrBasic)
            .

            DNDS_PY_DEF_SIMP_FUNC(ConstructFaceArea)
            .DNDS_PY_DEF_SIMP_FUNC(ConstructFaceCent)
            .DNDS_PY_DEF_SIMP_FUNC(ConstructFaceIntJacobiDet)
            .DNDS_PY_DEF_SIMP_FUNC(ConstructFaceIntPPhysics)
            .DNDS_PY_DEF_SIMP_FUNC(ConstructFaceUnitNorm)
            . // on quad int points
            DNDS_PY_DEF_SIMP_FUNC(ConstructFaceMeanNorm)
            .

            DNDS_PY_DEF_SIMP_FUNC(ConstructCellSmoothScale);

        FiniteVolume_
            .def("GetCellAtr", &T::GetCellAtr, py::arg("iCell"), py::return_value_policy::reference_internal)
            .def("GetCellOrder", &T::GetCellOrder, py::arg("iCell"));

        FiniteVolume_
            .def("GetFaceAtr", &T::GetFaceAtr, py::arg("iFace"), py::return_value_policy::reference_internal);

        FiniteVolume_
            .def("GetCellVol", &T::GetCellVol, py::arg("iCell"))
            .def("GetFaceArea", &T::GetFaceArea, py::arg("iFace"));

        FiniteVolume_
            .def("GetGlobalVol", &T::GetGlobalVol);
        FiniteVolume_
            .def("GetCellSmoothScaleRatio", &T::GetCellSmoothScaleRatio, py::arg("iCell"));

        FiniteVolume_
            .def("GetCellJacobiDet", &T::GetCellJacobiDet, py::arg("iCell"), py::arg("iG"))
            .def("GetFaceJacobiDet", &T::GetFaceJacobiDet, py::arg("iFace"), py::arg("iG"));

        FiniteVolume_
            .def("GetFaceParamArea", &T::GetFaceParamArea, py::arg("iFace"));

        FiniteVolume_
            .def("GetCellBary", &T::GetCellBary, py::arg("iCell"));

        FiniteVolume_
            .def("CellIsFaceBack", &T::CellIsFaceBack, py::arg("iCell"), py::arg("iFace"));
        FiniteVolume_
            .def("CellFaceOther", &T::CellFaceOther, py::arg("iCell"), py::arg("iFace"));

        FiniteVolume_
            .def("GetFaceNorm", &T::GetFaceNorm, py::arg("iFace"), py::arg("iG"));
        FiniteVolume_
            .def("GetFaceNormFromCell", &T::GetFaceNormFromCell, py::arg("iFace"), py::arg("iCell"), py::arg("if2c"), py::arg("iG"));

        FiniteVolume_
            .def("GetFaceQuadraturePPhys", &T::GetFaceQuadraturePPhys, py::arg("iFace"), py::arg("iG"));
        FiniteVolume_
            .def("GetFaceQuadraturePPhysFromCell", &T::GetFaceQuadraturePPhysFromCell, py::arg("iFace"), py::arg("iCell"), py::arg("if2c"), py::arg("iG"));
        FiniteVolume_
            .def("GetFacePointFromCell", &T::GetFacePointFromCell, py::arg("iFace"), py::arg("iCell"), py::arg("if2c"), py::arg("pnt"));

        FiniteVolume_
            .def("GetOtherCellBaryFromCell", &T::GetOtherCellBaryFromCell, py::arg("iCell"), py::arg("iCellOther"), py::arg("iFace"));
        FiniteVolume_
            .def("GetOtherCellPointFromCell", &T::GetOtherCellPointFromCell, py::arg("iCell"), py::arg("iCellOther"), py::arg("iFace"), py::arg("pnt"));
        FiniteVolume_
            .def("GetOtherCellInertiaFromCell", &T::GetOtherCellInertiaFromCell, py::arg("iCell"), py::arg("iCellOther"), py::arg("iFace"));
        FiniteVolume_
            .def("GetCellQuadraturePPhys", &T::GetCellQuadraturePPhys, py::arg("iCell"), py::arg("iG"));
        FiniteVolume_
            .def("GetCellMaxLenScale", &T::GetCellMaxLenScale, py::arg("iCell"));
        FiniteVolume_
            .def("getArrayBytes", &T::getArrayBytes);

        FiniteVolume_
            .def("to_device", [](FiniteVolume &self, const std::string &backend)
                 { self.to_device(device_backend_name_to_enum(backend)); }, py::arg("backend"))
            .def("to_host", &FiniteVolume::to_host);

#define DNDS_CFV_FV_PYBIND11_DEFINE_BuildUDof(nVarsFixed)                                                  \
    FiniteVolume_.def(                                                                                     \
        ("BuildUDof_" + RowSize_To_PySnippet(nVarsFixed)).c_str(),                                         \
        [](T &self, tUDof<nVarsFixed> &u, int nVars, bool buildSon, bool buildTrans, Geom::MeshLoc varloc) \
        { self.BuildUDof(u, nVars, buildSon, buildTrans, varloc); },                                               \
        py::arg("u"), py::arg("nVars"), py::arg("buildSon") = true,                                        \
        py::arg("buildTrans") = true, py::arg("varloc") = Geom::MeshLoc::Cell,                             \
        DNDS_PYBIND11_OSTREAM_GUARD)

#define DNDS_CFV_FV_PYBIND11_DEFINE_BuildUGrad(nVarsFixed)                                                         \
    FiniteVolume_                                                                                                  \
        .def(                                                                                                      \
            ("BuildUGrad_2x" + RowSize_To_PySnippet(nVarsFixed)).c_str(),                                          \
            [](T &self, tUGrad<nVarsFixed, 2> &u, int nVars, bool buildSon, bool buildTrans, Geom::MeshLoc varloc) \
            { self.BuildUGradD(u, nVars, buildSon, buildTrans, varloc); },                                         \
            py::arg("u"), py::arg("nVars"), py::arg("buildSon") = true,                                            \
            py::arg("buildTrans") = true, py::arg("varloc") = Geom::MeshLoc::Cell,                                 \
            DNDS_PYBIND11_OSTREAM_GUARD)                                                                           \
        .def(                                                                                                      \
            ("BuildUGrad_3x" + RowSize_To_PySnippet(nVarsFixed)).c_str(),                                          \
            [](T &self, tUGrad<nVarsFixed, 3> &u, int nVars, bool buildSon, bool buildTrans, Geom::MeshLoc varloc) \
            { self.BuildUGradD(u, nVars, buildSon, buildTrans, varloc); },                                         \
            py::arg("u"), py::arg("nVars"), py::arg("buildSon") = true,                                            \
            py::arg("buildTrans") = true, py::arg("varloc") = Geom::MeshLoc::Cell,                                 \
            DNDS_PYBIND11_OSTREAM_GUARD)

#define DNDS_CFV_FV_PYBIND11_DEFINE_BuildCalls(nVarsFixed)  \
    {                                                       \
        DNDS_CFV_FV_PYBIND11_DEFINE_BuildUDof(nVarsFixed);  \
        DNDS_CFV_FV_PYBIND11_DEFINE_BuildUGrad(nVarsFixed); \
    }

        DNDS_CFV_FV_PYBIND11_DEFINE_BuildCalls(1);
        DNDS_CFV_FV_PYBIND11_DEFINE_BuildCalls(4);
        DNDS_CFV_FV_PYBIND11_DEFINE_BuildCalls(5);
        DNDS_CFV_FV_PYBIND11_DEFINE_BuildCalls(6);
        DNDS_CFV_FV_PYBIND11_DEFINE_BuildCalls(7);
        DNDS_CFV_FV_PYBIND11_DEFINE_BuildCalls(DynamicSize);
    }
}