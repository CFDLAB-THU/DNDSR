#pragma once

#include "DNDS/Defines_bind.hpp"
#include "DNDS/Array_bind.hpp"

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
            .def("ConstructBaseAndWeight", [&](T &self)
                 { self.ConstructBaseAndWeight(); })
            // todo: the functor-in version; remember the GIL!
            .def("ConstructRecCoeff", &T::ConstructRecCoeff);
        // TODO: wrap Euler-related calls in EulerSolver inside euler!

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
            DNDS_CFV_VR_PYBIND11_DEFINE_BuildCalls(4);
            DNDS_CFV_VR_PYBIND11_DEFINE_BuildCalls(5);
            DNDS_CFV_VR_PYBIND11_DEFINE_BuildCalls(6);
            DNDS_CFV_VR_PYBIND11_DEFINE_BuildCalls(7);
            DNDS_CFV_VR_PYBIND11_DEFINE_BuildCalls(DynamicSize);
        }
        else
        {
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