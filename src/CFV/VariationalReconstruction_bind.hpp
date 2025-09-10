#pragma once

#include "DNDS/Defines_bind.hpp"
#include "DNDS/Array_bind.hpp"
#include <pybind11/functional.h>
#include <pybind11_json/pybind11_json.hpp>
#include <pybind11/eigen.h>

#include "VariationalReconstruction.hpp"

namespace DNDS::CFV
{
    template <int dim = 2>
    using tPy_VariationalReconstruction = py_class_ssp<VariationalReconstruction<dim>>;

    template <int dim = 2>
    // static const int dim = 2;
    void pybind11_VariationalReconstruction_define(py::module_ &m)
    {
        auto VariationalReconstruction_ = tPy_VariationalReconstruction<dim>(m, fmt::format("VariationalReconstruction_{}", dim).c_str());
        using T = VariationalReconstruction<dim>;
        VariationalReconstruction_
            .def(py::init([](MPIInfo &mpi, ssp<Geom::UnstructuredMesh> mesh)
                          { return std::make_shared<T>(mpi, mesh); }),
                 py::arg("mpi"), py::arg("mesh"))
            .def("ConstructMetrics", &T::ConstructMetrics)
            .def(
                "ConstructBaseAndWeight",
                [](T &self, typename T::tFGetBoundaryWeight f)
                {
                    self.ConstructBaseAndWeight(
                        [f](Geom::t_index id, int order)
                        {
                            py::gil_scoped_acquire scope_gil;
                            return f(id, order);
                        });
                },
                py::arg("map_bcId_iOrder_to_bCweight"))
            .def(
                "ConstructBaseAndWeight_map",
                [](T &self, const std::map<std::pair<Geom::t_index, int>, real> &m)
                {
                    self.ConstructBaseAndWeight(
                        [&](Geom::t_index id, int order)
                        {
                            if (m.count({id, order}))
                                return m.at({id, order});
                            else
                                return 0.0;
                        });
                },
                py::arg("map_bcId_iOrder_to_bCweight"))
            .def("ConstructRecCoeff", &T::ConstructRecCoeff);
        // TODO: wrap Euler-related calls in EulerSolver inside euler!

        VariationalReconstruction_
            .def("SetPeriodicTransformations3d", [](T &self, std::array<int, 3> Seq123)
                 { self.SetPeriodicTransformations(Seq123); }, py::arg("Seq123"))
            .def("SetPeriodicTransformations2d", [](T &self, std::array<int, 2> Seq123)
                 { self.SetPeriodicTransformations(Seq123); }, py::arg("Seq123"))
            .def("SetPeriodicTransformationsNoOp", [](T &self)
                 { self.SetPeriodicTransformations(); });

        VariationalReconstruction_
            .def(
                "ParseSettings", [](T &self, py::object settings)
                { 
                    VRSettings defaultSettings(self.getDim());
                    nlohmann::ordered_json defaultJson;
                    defaultSettings.WriteIntoJson(defaultJson);
                    nlohmann::json settings_json = settings;
                    defaultJson.merge_patch(settings_json);
                    self.settings.ParseFromJson(defaultJson); },
                py::arg("Seq123"));
        VariationalReconstruction_
            .def(
                "GetCellBary", &T::GetCellBary, py::arg("iCell"));

#define DNDS_CFV_VR_PYBIND11_DEFINE_BuildUDof(nVarsFixed)                            \
    VariationalReconstruction_.def(                                                  \
        ("BuildUDof_" + RowSize_To_PySnippet(nVarsFixed)).c_str(),                   \
        [](T &self, tUDof<nVarsFixed> &u, int nVars, bool buildSon, bool buildTrans) \
        { self.BuildUDof(u, nVars, buildSon, buildTrans); },                         \
        py::arg("u"), py::arg("nVars"), py::arg("buildSon") = true, py::arg("buildTrans") = true)
#define DNDS_CFV_VR_PYBIND11_DEFINE_BuildURec(nVarsFixed)                            \
    VariationalReconstruction_.def(                                                  \
        ("BuildURec_" + RowSize_To_PySnippet(nVarsFixed)).c_str(),                   \
        [](T &self, tURec<nVarsFixed> &u, int nVars, bool buildSon, bool buildTrans) \
        { self.BuildURec(u, nVars, buildSon, buildTrans); },                         \
        py::arg("u"), py::arg("nVars"), py::arg("buildSon") = true, py::arg("buildTrans") = true)
#define DNDS_CFV_VR_PYBIND11_DEFINE_BuildUGrad(nVarsFixed)                                 \
    VariationalReconstruction_.def(                                                        \
        ("BuildUGrad_" + RowSize_To_PySnippet(nVarsFixed)).c_str(),                        \
        [](T &self, tUGrad<nVarsFixed, dim> &u, int nVars, bool buildSon, bool buildTrans) \
        { self.BuildUGrad(u, nVars, buildSon, buildTrans); },                              \
        py::arg("u"), py::arg("nVars"), py::arg("buildSon") = true, py::arg("buildTrans") = true)
#define DNDS_CFV_VR_PYBIND11_DEFINE_BuildCalls(nVarsFixed)  \
    {                                                       \
        DNDS_CFV_VR_PYBIND11_DEFINE_BuildUDof(nVarsFixed);  \
        DNDS_CFV_VR_PYBIND11_DEFINE_BuildURec(nVarsFixed);  \
        DNDS_CFV_VR_PYBIND11_DEFINE_BuildUGrad(nVarsFixed); \
    }
        if constexpr (dim == 2)
        {
            DNDS_CFV_VR_PYBIND11_DEFINE_BuildCalls(1);
            DNDS_CFV_VR_PYBIND11_DEFINE_BuildCalls(4);
            DNDS_CFV_VR_PYBIND11_DEFINE_BuildCalls(5);
            DNDS_CFV_VR_PYBIND11_DEFINE_BuildCalls(6);
            DNDS_CFV_VR_PYBIND11_DEFINE_BuildCalls(7);
            DNDS_CFV_VR_PYBIND11_DEFINE_BuildCalls(DynamicSize);
        }
        else
        {
            DNDS_CFV_VR_PYBIND11_DEFINE_BuildCalls(1);
            DNDS_CFV_VR_PYBIND11_DEFINE_BuildCalls(5);
            DNDS_CFV_VR_PYBIND11_DEFINE_BuildCalls(6);
            DNDS_CFV_VR_PYBIND11_DEFINE_BuildCalls(7);
            DNDS_CFV_VR_PYBIND11_DEFINE_BuildCalls(DynamicSize);
        }
#undef DNDS_CFV_VR_PYBIND11_DEFINE_BuildUDof
#undef DNDS_CFV_VR_PYBIND11_DEFINE_BuildURec
#undef DNDS_CFV_VR_PYBIND11_DEFINE_BuildUGrad
#undef DNDS_CFV_VR_PYBIND11_DEFINE_BuildCalls
    }
}