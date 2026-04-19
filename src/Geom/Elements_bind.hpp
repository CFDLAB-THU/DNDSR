#pragma once

#include "DNDS/Defines_bind.hpp"

#include "Elements.hpp"

namespace DNDS::Geom::Elem
{

    using tPy_ElemType = py::enum_<ElemType>;

    inline void pybind11_ElemType_define(py::module_ &m)
    {
        tPy_ElemType(m, "ElemType")
            .value("UnknownElem", ElemType::UnknownElem)
            .value("Line2", ElemType::Line2)
            .value("Line3", ElemType::Line3)
            .value("Tri3", ElemType::Tri3)
            .value("Tri6", ElemType::Tri6)
            .value("Quad4", ElemType::Quad4)
            .value("Quad9", ElemType::Quad9)
            .value("Tet4", ElemType::Tet4)
            .value("Tet10", ElemType::Tet10)
            .value("Hex8", ElemType::Hex8)
            .value("Hex27", ElemType::Hex27)
            .value("Prism6", ElemType::Prism6)
            .value("Prism18", ElemType::Prism18)
            .value("Pyramid5", ElemType::Pyramid5)
            .value("Pyramid14", ElemType::Pyramid14);
    }
}