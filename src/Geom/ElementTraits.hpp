#pragma once
// ElementTraits.hpp -- Per-element-type trait structs with static dispatch.
//
// This is the HUB header that includes all element trait definitions.
// Each element type's traits are defined in a separate file under Elements/:
//   - Line2, Line3  -> Elements/Line2.hpp, Elements/Line3.hpp
//   - Tri3, Tri6    -> Elements/Tri3.hpp, Elements/Tri6.hpp
//   - Quad4, Quad9  -> Elements/Quad4.hpp, Elements/Quad9.hpp
//   - Tet4, Tet10   -> Elements/Tet4.hpp, Elements/Tet10.hpp
//   - Hex8, Hex27   -> Elements/Hex8.hpp, Elements/Hex27.hpp
//   - Prism6, Prism18 -> Elements/Prism6.hpp, Elements/Prism18.hpp
//   - Pyramid5, Pyramid14 -> Elements/Pyramid5.hpp, Elements/Pyramid14.hpp
//
// To add a new element type:
//   1. Add its enum value to ElemType in ElemEnum.hpp
//   2. Create an ElementTraits specialization in Elements/<Elem>.hpp
//   3. Add its case to DispatchElementType() below
//   4. Add its shape function definition in tools/gen_shape_functions/

#include "ElementTraitsBase.hpp"

// Include all element trait specializations
// Note: These files contain both ShapeFuncImpl (generated) and ElementTraits (manual)
#include "Elements/Line2.hpp"
#include "Elements/Line3.hpp"
#include "Elements/Tri3.hpp"
#include "Elements/Tri6.hpp"
#include "Elements/Quad4.hpp"
#include "Elements/Quad9.hpp"
#include "Elements/Tet4.hpp"
#include "Elements/Tet10.hpp"
#include "Elements/Hex8.hpp"
#include "Elements/Hex27.hpp"
#include "Elements/Prism6.hpp"
#include "Elements/Prism18.hpp"
#include "Elements/Pyramid5.hpp"
#include "Elements/Pyramid14.hpp"

namespace DNDS::Geom::Elem
{

    // ----------------------------------------------------------------
    // ParamSpace-level information (not per-element, per-space)
    // ----------------------------------------------------------------
    
    /**
     * @brief Get the volume of a parametric space
     * @param ps Parametric space type
     * @return Volume (length for 1D, area for 2D, volume for 3D)
     * 
     * Returns the "size" of the reference element in parametric coordinates:
     * - LineSpace:    2.0   (length from -1 to +1)
     * - TriSpace:     0.5   (area of reference triangle)
     * - QuadSpace:    4.0   (area of [-1,1] x [-1,1])
     * - TetSpace:     1/6   (volume of reference tet)
     * - HexSpace:     8.0   (volume of [-1,1]^3)
     * - PrismSpace:   1.0   (volume of reference prism)
     * - PyramidSpace: 4/3   (volume of reference pyramid)
     */
    DNDS_DEVICE_CALLABLE constexpr t_real ParamSpaceVolume(ParamSpace ps)
    {
        switch (ps)
        {
        case LineSpace:    return 2.0;
        case TriSpace:     return 0.5;
        case QuadSpace:    return 4.0;
        case TetSpace:     return 1.0 / 6.0;
        case HexSpace:     return 8.0;
        case PrismSpace:   return 1.0;
        case PyramidSpace: return 4.0 / 3.0;
        default:           return 0.0;
        }
    }

    /**
     * @brief Get the order-1 (linear) element type for a parametric space
     * @param ps Parametric space type
     * @return Element type enum (Line2, Tri3, Quad4, etc.)
     * 
     * Maps each parametric space to its corresponding linear element:
     * - LineSpace    -> Line2
     * - TriSpace     -> Tri3
     * - QuadSpace    -> Quad4
     * - TetSpace     -> Tet4
     * - HexSpace     -> Hex8
     * - PrismSpace   -> Prism6
     * - PyramidSpace -> Pyramid5
     */
    DNDS_DEVICE_CALLABLE constexpr ElemType ParamSpaceO1Elem(ParamSpace ps)
    {
        switch (ps)
        {
        case LineSpace:    return Line2;
        case TriSpace:     return Tri3;
        case QuadSpace:    return Quad4;
        case TetSpace:     return Tet4;
        case HexSpace:     return Hex8;
        case PrismSpace:   return Prism6;
        case PyramidSpace: return Pyramid5;
        default:           return UnknownElem;
        }
    }

    // ----------------------------------------------------------------
    // Compile-time dispatch: invoke a callable with ElementTraits<t>
    // ----------------------------------------------------------------
    
    /**
     * @brief Static dispatch over element types at compile time
     * @tparam Func Callable type
     * @param t   Element type enum
     * @param func Callable to invoke with ElementTraits<t>
     * @return Return value of func(ElementTraits<t>{})
     * 
     * Usage:
     *   auto numNodes = DispatchElementType(elemType, [](auto traits) {
     *       return traits.numNodes;
     *   });
     */
    template <typename Func>
    DNDS_DEVICE_CALLABLE constexpr decltype(auto) DispatchElementType(ElemType t, Func &&func)
    {
        switch (t)
        {
        case Line2:
            return func(ElementTraits<Line2>{});
        case Line3:
            return func(ElementTraits<Line3>{});
        case Tri3:
            return func(ElementTraits<Tri3>{});
        case Tri6:
            return func(ElementTraits<Tri6>{});
        case Quad4:
            return func(ElementTraits<Quad4>{});
        case Quad9:
            return func(ElementTraits<Quad9>{});
        case Tet4:
            return func(ElementTraits<Tet4>{});
        case Tet10:
            return func(ElementTraits<Tet10>{});
        case Hex8:
            return func(ElementTraits<Hex8>{});
        case Hex27:
            return func(ElementTraits<Hex27>{});
        case Prism6:
            return func(ElementTraits<Prism6>{});
        case Prism18:
            return func(ElementTraits<Prism18>{});
        case Pyramid5:
            return func(ElementTraits<Pyramid5>{});
        case Pyramid14:
            return func(ElementTraits<Pyramid14>{});
        default:
            DNDS_assert(false);
            return func(ElementTraits<Line2>{}); // unreachable, satisfies return type
        }
    }

} // namespace DNDS::Geom::Elem
