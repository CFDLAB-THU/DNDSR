#pragma once
// ElemEnum.hpp -- Element type and parameter space enumerations.
//
// Extracted from Elements.hpp so that ElementTraits.hpp and other headers
// can use the enums without pulling in the full Elements definition.

#include <cstdint>

namespace DNDS::Geom::Elem
{
    /**
     * Complying to [CGNS Element standard](https://cgns.github.io/CGNS_docs_current/sids/conv.html)
     *  !note that we use 0 based indexing (CGNS uses 1 based in the link)
     */

    static const int CellNumNodeMax = 27;

    enum ElemType
    {
        UnknownElem = 0,
        Line2 = 1,
        Line3 = 8,

        Tri3 = 2,
        Tri6 = 9,
        Quad4 = 3,
        Quad9 = 10,

        Tet4 = 4,
        Tet10 = 11,
        Hex8 = 5,
        Hex27 = 12,
        Prism6 = 6,
        Prism18 = 13,
        Pyramid5 = 7,
        Pyramid14 = 14,

        ElemType_NUM = 15
    };

    enum ParamSpace
    {
        UnknownPSpace = 0,
        LineSpace = 1,

        TriSpace = 2,
        QuadSpace = 3,

        TetSpace = 4,
        HexSpace = 5,
        PrismSpace = 6,
        PyramidSpace = 7,

        ParamSpace_NUM = 8
    };

} // namespace DNDS::Geom::Elem
