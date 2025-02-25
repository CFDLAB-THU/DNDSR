#pragma once

#include "Elements.hpp"

#include <cgnslib.h>

namespace DNDS::Geom
{
#define DNDS_CGNS_CALL_EXIT(call) {if (call) cg_error_exit();}

    // todo: make these mappings easy to maintain (like, inverted with program)
    inline constexpr Elem::ElemType __getElemTypeFromCGNSType(ElementType_t cgns_et)
    {
        switch (cgns_et)
        {
        case BAR_2:
            return Elem::Line2;
        case BAR_3:
            return Elem::Line3;
        case TRI_3:
            return Elem::Tri3;
        case TRI_6:
        case QUAD_4:
            return Elem::Quad4;
        case QUAD_9:
            return Elem::Quad9;
        case TETRA_4:
            return Elem::Tet4;
        case TETRA_10:
            return Elem::Tet10;
        case HEXA_8:
            return Elem::Hex8;
        case HEXA_27:
            return Elem::Hex27;
        case PENTA_6:
            return Elem::Prism6;
        case PENTA_18:
            return Elem::Prism18;
        case PYRA_5:
            return Elem::Pyramid5;
        case PYRA_14:
            return Elem::Pyramid14;
        default:
            return Elem::UnknownElem;
        }
    }

    inline constexpr ElementType_t __getCGNSTypeFromElemType(Elem::ElemType et)
    {
        switch (et)
        {
        case Elem::Line2:
            return BAR_2;
        case Elem::Line3:
            return BAR_3;
        case Elem::Tri3:
            return TRI_3;
        case Elem::Tri6:
            return TRI_6;
        case Elem::Quad4:
            return QUAD_4;
        case Elem::Quad9:
            return QUAD_9;
        case Elem::Tet4:
            return TETRA_4;
        case Elem::Tet10:
            return TETRA_10;
        case Elem::Hex8:
            return HEXA_8;
        case Elem::Hex27:
            return HEXA_27;
        case Elem::Prism6:
            return PENTA_6;
        case Elem::Prism18:
            return PENTA_18;
        case Elem::Pyramid5:
            return PYRA_5;
        case Elem::Pyramid14:
            return PYRA_14;
        default:
            return ElementTypeNull;
        }
    }

}