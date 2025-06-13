#pragma once

#include "DNDS/Defines_bind.hpp"
#include "DNDS/Array_bind.hpp"
#include "DNDS/ArrayDerived/ArrayEigenMatrix_bind.hpp"
#include "VRDefines.hpp"

namespace DNDS::CFV
{
    // is alias really necessary?
    template <int nVarsFixed>
    inline void pybind11_define_tUDof_alias(py::module_ &m, py::module_ &m_dnds)
    {
        m.attr(("tUDof_" + RowSize_To_PySnippet(nVarsFixed)).c_str()) =
            pybind11_ArrayEigenMatrixPair_get_class<nVarsFixed, 1>(m_dnds);
    }

    inline void pybind11_VRDefines_define(py::module_ &m, py::module_ &m_dnds)
    {
#define DNDS_CFV_DEFINE_ALIAS(nVarsFixed)                   \
    {                                                       \
        pybind11_define_tUDof_alias<nVarsFixed>(m, m_dnds); \
    }
        DNDS_CFV_DEFINE_ALIAS(1);
        DNDS_CFV_DEFINE_ALIAS(2);
        DNDS_CFV_DEFINE_ALIAS(3);
        DNDS_CFV_DEFINE_ALIAS(4);
        DNDS_CFV_DEFINE_ALIAS(5);
        DNDS_CFV_DEFINE_ALIAS(6);
        DNDS_CFV_DEFINE_ALIAS(7);
        DNDS_CFV_DEFINE_ALIAS(DynamicSize);
    }
}