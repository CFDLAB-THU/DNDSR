#pragma once

#include "DNDS/Defines_bind.hpp"
#include "DNDS/Array_bind.hpp"
#include "DNDS/ArrayDerived/ArrayEigenMatrix_bind.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch_bind.hpp"
#include "DNDS/ArrayDerived/ArrayEigenVector_bind.hpp"
#include "VRDefines.hpp"
#include "DNDS/ArrayDOF_bind.hpp"

namespace DNDS::CFV
{
    // is alias really necessary?
    template <int nVarsFixed>
    inline void pybind11_define_tURec_alias(py::module_ &m, py::module_ &m_dnds)
    {
        m.attr(("tURec_" + RowSize_To_PySnippet(nVarsFixed)).c_str()) =
            pybind11_ArrayDOF_get_class<DynamicSize, nVarsFixed>(m_dnds);
    }

    template <int nVarsFixed>
    inline void pybind11_define_tUDof_alias(py::module_ &m, py::module_ &m_dnds)
    {
        m.attr(("tUDof_" + RowSize_To_PySnippet(nVarsFixed)).c_str()) =
            pybind11_ArrayDOF_get_class<nVarsFixed, 1>(m_dnds);
    }

    template <int nVarsFixed>
    inline void pybind11_define_tUGrad_alias(py::module_ &m, py::module_ &m_dnds)
    {
        m.attr(("tUGrad_2x" + RowSize_To_PySnippet(nVarsFixed)).c_str()) =
            pybind11_ArrayDOF_get_class<2, nVarsFixed>(m_dnds);
        m.attr(("tUGrad_3x" + RowSize_To_PySnippet(nVarsFixed)).c_str()) =
            pybind11_ArrayDOF_get_class<3, nVarsFixed>(m_dnds);
    }

    inline void pybind11_define_tVVecPair_alias(py::module_ &m, py::module_ &m_dnds)
    {
        m.attr("tVVecPair") =
            pybind11_ArrayEigenVectorPair_get_class<DynamicSize, DynamicSize>(m_dnds);
    }

    inline void pybind11_define_tMatsPair_alias(py::module_ &m, py::module_ &m_dnds)
    {
        m.attr("tMatsPair") =
            pybind11_ArrayEigenUniMatrixBatchPair_get_class<DynamicSize, DynamicSize>(m_dnds);
    }

    using tPy_RecAtr = py::class_<RecAtr>;
    inline void pybind11_define_RecAtr(py::module_ &m)
    {
        auto RecAtr_ = tPy_RecAtr(m, "RecAtr");
        RecAtr_.def(py::init());
        RecAtr_
            .def_readwrite("intOrder", &RecAtr::intOrder)
            .def_readwrite("NDIFF", &RecAtr::NDIFF)
            .def_readwrite("NDOF", &RecAtr::NDOF)
            .def_readwrite("Order", &RecAtr::Order)
            .def_readwrite("relax", &RecAtr::relax);
    }

    inline void pybind11_define_RecAtrArrayPair_and_alias(py::module_ &m)
    {
        pybind11_Array_define<RecAtr>(m);
        pybind11_ParArray_define<RecAtr>(m);
        pybind11_ArrayTransformer_define<ParArray<RecAtr>>(m);
        pybind11_ParArrayPair_define<RecAtr>(m);
        m.attr("tRecAtrPair") =
            pybind11_ParArrayPair_get_class<RecAtr>(m);
    }

    // TODO: tVecsPair, tVMatPair, tCoeffPair, t3VecsPair, t3VecPair, t3MatPair

    inline void pybind11_VRDefines_define(py::module_ &m, py::module_ &m_dnds)
    {
#define DNDS_CFV_DEFINE_ALIAS(nVarsFixed)                    \
    {                                                        \
        pybind11_define_tURec_alias<nVarsFixed>(m, m_dnds);  \
        pybind11_define_tUDof_alias<nVarsFixed>(m, m_dnds);  \
        pybind11_define_tUGrad_alias<nVarsFixed>(m, m_dnds); \
    }
        DNDS_CFV_DEFINE_ALIAS(1);
        DNDS_CFV_DEFINE_ALIAS(2);
        DNDS_CFV_DEFINE_ALIAS(3);
        DNDS_CFV_DEFINE_ALIAS(4);
        DNDS_CFV_DEFINE_ALIAS(5);
        DNDS_CFV_DEFINE_ALIAS(6);
        DNDS_CFV_DEFINE_ALIAS(7);
        DNDS_CFV_DEFINE_ALIAS(DynamicSize);
#undef DNDS_CFV_DEFINE_ALIAS

        pybind11_define_tVVecPair_alias(m, m_dnds);
        pybind11_define_tMatsPair_alias(m, m_dnds);
        pybind11_define_RecAtr(m);
        pybind11_define_RecAtrArrayPair_and_alias(m);
    }
}