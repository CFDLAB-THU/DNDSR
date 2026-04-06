#pragma once
// ElementTraits.hpp -- Per-element-type trait structs with static dispatch.
//
// Each element family (Line, Tri, Quad, Tet, Hex, Prism, Pyramid) is defined
// by a traits struct that concentrates all metadata and topology for that type.
// Shape functions are provided via generated code in Geom/Elements/<Elem>.hpp.
//
// To add a new element type:
//   1. Add its enum value to ElemType in ElemEnum.hpp
//   2. Create an ElementTraits specialization following the pattern below
//   3. Add its case to DispatchElementType()
//   4. Add its shape function definition in tools/gen_shape_functions/

#include <array>
#include <cstdint>
#include "DNDS/Defines.hpp"
#include "Geometric.hpp"
#include "ElemEnum.hpp"

namespace DNDS::Geom::Elem
{

    // ----------------------------------------------------------------
    // ElementTraits<ElemType t> -- primary template (unspecialized = error)
    // ----------------------------------------------------------------
    template <ElemType t>
    struct ElementTraits; // must be specialized for each valid ElemType

    // ----------------------------------------------------------------
    // Macro to reduce boilerplate in each specialization
    // ----------------------------------------------------------------
#define DNDS_ELEMENT_TRAITS_COMMON(ETYPE, DIM_, ORDER_, NV_, NN_, NF_, PSPACE_, PSVOL_) \
    static constexpr ElemType elemType = ETYPE;                                          \
    static constexpr int dim = DIM_;                                                     \
    static constexpr int order = ORDER_;                                                 \
    static constexpr int numVertices = NV_;                                              \
    static constexpr int numNodes = NN_;                                                 \
    static constexpr int numFaces = NF_;                                                 \
    static constexpr ParamSpace paramSpace = PSPACE_;                                    \
    static constexpr t_real paramSpaceVol = PSVOL_;

    // ----------------------------------------------------------------
    // Max elevation span width (Hex8 body center uses 8 nodes)
    // ----------------------------------------------------------------
    static constexpr int elevSpanMaxWidth = 8;
    using tElevSpan = std::array<t_index, elevSpanMaxWidth>;

    // ----------------------------------------------------------------
    // Max sub-element node count for bisection (Hex8 = 8 nodes)
    // ----------------------------------------------------------------
    static constexpr int bisectSubMaxNodes = 8;
    using tBisectSub = std::array<t_index, bisectSubMaxNodes>;

    // ----------------------------------------------------------------
    // VTK node order: max 27 (Hex27); only first numNodes entries used,
    // and only entries that differ from identity are interesting.
    // We store the full permutation for simplicity.
    // ----------------------------------------------------------------
    static constexpr int vtkNodeOrderMax = 27;
    using tVTKNodeOrder = std::array<int, vtkNodeOrderMax>;

    // ----------------------------------------------------------------
    // 1D Elements
    // ----------------------------------------------------------------
    template <>
    struct ElementTraits<Line2>
    {
        DNDS_ELEMENT_TRAITS_COMMON(Line2, 1, 1, 2, 2, 0, LineSpace, 2.0)

        static constexpr std::array<t_real, 3 * 2> standardCoords = {
            -1, 0, 0,
             1, 0, 0};

        static constexpr ElemType GetFaceType(t_index /*iFace*/) { return UnknownElem; }

        // Elevation O1->O2
        static constexpr ElemType elevatedType = Line3;
        static constexpr int numElevNodes = 1;
        static constexpr std::array<tElevSpan, 1> elevSpans = {{
            {0, 1}}};
        static constexpr std::array<ElemType, 1> elevNodeSpanTypes = {Line2};

        // VTK
        static constexpr int vtkCellType = 3;
        static constexpr std::array<int, 2> vtkNodeOrder = {0, 1};
    };

    template <>
    struct ElementTraits<Line3>
    {
        DNDS_ELEMENT_TRAITS_COMMON(Line3, 1, 2, 2, 3, 0, LineSpace, 2.0)

        static constexpr std::array<t_real, 3 * 3> standardCoords = {
            -1, 0, 0,
             1, 0, 0,
             0, 0, 0};

        static constexpr ElemType GetFaceType(t_index /*iFace*/) { return UnknownElem; }
        static constexpr ElemType elevatedType = UnknownElem;
        static constexpr int numElevNodes = 0;

        // O2 bisect
        static constexpr int numBisect = 2;
        static constexpr int numBisectVariants = 1;
        static constexpr ElemType GetBisectElemType(t_index /*i*/) { return Line2; }
        static constexpr std::array<tBisectSub, 2> bisectElements = {{
            {0, 2},
            {2, 1}}};

        // VTK
        static constexpr int vtkCellType = 4;
        static constexpr std::array<int, 3> vtkNodeOrder = {0, 2, 1};
    };

    // ----------------------------------------------------------------
    // 2D Elements
    // ----------------------------------------------------------------
    template <>
    struct ElementTraits<Tri3>
    {
        DNDS_ELEMENT_TRAITS_COMMON(Tri3, 2, 1, 3, 3, 3, TriSpace, 0.5)

        static constexpr std::array<t_real, 3 * 3> standardCoords = {
            0, 0, 0,
            1, 0, 0,
            0, 1, 0};

        static constexpr ElemType GetFaceType(t_index /*iFace*/) { return Line2; }

        static constexpr std::array<std::array<t_index, 10>, 3> faceNodes = {{
            {0, 1},
            {1, 2},
            {2, 0}}};

        static constexpr ElemType elevatedType = Tri6;
        static constexpr int numElevNodes = 3;
        static constexpr std::array<tElevSpan, 3> elevSpans = {{
            {0, 1}, {1, 2}, {2, 0}}};
        static constexpr std::array<ElemType, 3> elevNodeSpanTypes = {Line2, Line2, Line2};

        // VTK
        static constexpr int vtkCellType = 5;
        static constexpr std::array<int, 3> vtkNodeOrder = {0, 1, 2};
    };

    template <>
    struct ElementTraits<Tri6>
    {
        DNDS_ELEMENT_TRAITS_COMMON(Tri6, 2, 2, 3, 6, 3, TriSpace, 0.5)

        static constexpr std::array<t_real, 3 * 6> standardCoords = {
            0, 0, 0,
            1, 0, 0,
            0, 1, 0,
            0.5, 0, 0,
            0.5, 0.5, 0,
            0, 0.5, 0};

        static constexpr ElemType GetFaceType(t_index /*iFace*/) { return Line3; }

        static constexpr std::array<std::array<t_index, 10>, 3> faceNodes = {{
            {0, 1, 3},
            {1, 2, 4},
            {2, 0, 5}}};

        static constexpr ElemType elevatedType = UnknownElem;
        static constexpr int numElevNodes = 0;

        // O2 bisect
        static constexpr int numBisect = 4;
        static constexpr int numBisectVariants = 1;
        static constexpr ElemType GetBisectElemType(t_index /*i*/) { return Tri3; }
        static constexpr std::array<tBisectSub, 4> bisectElements = {{
            {0, 3, 5},
            {3, 1, 4},
            {5, 3, 4},
            {5, 4, 2}}};

        // VTK
        static constexpr int vtkCellType = 22;
        static constexpr std::array<int, 6> vtkNodeOrder = {0, 1, 2, 3, 4, 5};
    };

    template <>
    struct ElementTraits<Quad4>
    {
        DNDS_ELEMENT_TRAITS_COMMON(Quad4, 2, 1, 4, 4, 4, QuadSpace, 4.0)

        static constexpr std::array<t_real, 3 * 4> standardCoords = {
            -1, -1, 0,
             1, -1, 0,
             1,  1, 0,
            -1,  1, 0};

        static constexpr ElemType GetFaceType(t_index /*iFace*/) { return Line2; }

        static constexpr std::array<std::array<t_index, 10>, 4> faceNodes = {{
            {0, 1},
            {1, 2},
            {2, 3},
            {3, 0}}};

        static constexpr ElemType elevatedType = Quad9;
        static constexpr int numElevNodes = 5;
        static constexpr std::array<tElevSpan, 5> elevSpans = {{
            {0, 1},
            {1, 2},
            {2, 3},
            {3, 0},
            {0, 1, 2, 3}}};
        static constexpr std::array<ElemType, 5> elevNodeSpanTypes = {
            Line2, Line2, Line2, Line2, Quad4};

        // VTK
        static constexpr int vtkCellType = 9;
        static constexpr std::array<int, 4> vtkNodeOrder = {0, 1, 2, 3};
    };

    template <>
    struct ElementTraits<Quad9>
    {
        DNDS_ELEMENT_TRAITS_COMMON(Quad9, 2, 2, 4, 9, 4, QuadSpace, 4.0)

        static constexpr std::array<t_real, 3 * 9> standardCoords = {
            -1, -1, 0,
             1, -1, 0,
             1,  1, 0,
            -1,  1, 0,
             0, -1, 0,
             1,  0, 0,
             0,  1, 0,
            -1,  0, 0,
             0,  0, 0};

        static constexpr ElemType GetFaceType(t_index /*iFace*/) { return Line3; }

        static constexpr std::array<std::array<t_index, 10>, 4> faceNodes = {{
            {0, 1, 4},
            {1, 2, 5},
            {2, 3, 6},
            {3, 0, 7}}};

        static constexpr ElemType elevatedType = UnknownElem;
        static constexpr int numElevNodes = 0;

        // O2 bisect
        static constexpr int numBisect = 4;
        static constexpr int numBisectVariants = 1;
        static constexpr ElemType GetBisectElemType(t_index /*i*/) { return Quad4; }
        static constexpr std::array<tBisectSub, 4> bisectElements = {{
            {0, 4, 8, 7},
            {4, 1, 5, 8},
            {7, 8, 6, 3},
            {8, 5, 2, 6}}};

        // VTK
        static constexpr int vtkCellType = 23;
        static constexpr std::array<int, 8> vtkNodeOrder = {0, 1, 2, 3, 4, 5, 6, 7};
    };

    // ----------------------------------------------------------------
    // 3D Elements
    // ----------------------------------------------------------------
    template <>
    struct ElementTraits<Tet4>
    {
        DNDS_ELEMENT_TRAITS_COMMON(Tet4, 3, 1, 4, 4, 4, TetSpace, 1.0 / 6.0)

        static constexpr std::array<t_real, 3 * 4> standardCoords = {
            0, 0, 0,
            1, 0, 0,
            0, 1, 0,
            0, 0, 1};

        static constexpr ElemType GetFaceType(t_index /*iFace*/) { return Tri3; }

        static constexpr std::array<std::array<t_index, 10>, 4> faceNodes = {{
            {0, 2, 1},
            {0, 1, 3},
            {1, 2, 3},
            {2, 0, 3}}};

        static constexpr ElemType elevatedType = Tet10;
        static constexpr int numElevNodes = 6;
        static constexpr std::array<tElevSpan, 6> elevSpans = {{
            {0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}}};
        static constexpr std::array<ElemType, 6> elevNodeSpanTypes = {
            Line2, Line2, Line2, Line2, Line2, Line2};

        // VTK
        static constexpr int vtkCellType = 10;
        static constexpr std::array<int, 4> vtkNodeOrder = {0, 1, 2, 3};
    };

    template <>
    struct ElementTraits<Tet10>
    {
        DNDS_ELEMENT_TRAITS_COMMON(Tet10, 3, 2, 4, 10, 4, TetSpace, 1.0 / 6.0)

        static constexpr std::array<t_real, 3 * 10> standardCoords = {
            0, 0, 0,
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
            0.5, 0, 0,
            0.5, 0.5, 0,
            0, 0.5, 0,
            0, 0, 0.5,
            0.5, 0, 0.5,
            0, 0.5, 0.5};

        static constexpr ElemType GetFaceType(t_index /*iFace*/) { return Tri6; }

        static constexpr std::array<std::array<t_index, 10>, 4> faceNodes = {{
            {0, 2, 1, 6, 5, 4},
            {0, 1, 3, 4, 8, 7},
            {1, 2, 3, 5, 9, 8},
            {2, 0, 3, 6, 7, 9}}};

        static constexpr ElemType elevatedType = UnknownElem;
        static constexpr int numElevNodes = 0;

        // O2 bisect: 8 sub-tets, 3 variants
        static constexpr int numBisect = 8;
        static constexpr int numBisectVariants = 3;
        static constexpr ElemType GetBisectElemType(t_index /*i*/) { return Tet4; }
        // Variant 0 (8), variant 1 (8), variant 2 (8) = 24 rows
        static constexpr std::array<tBisectSub, 24> bisectElements = {{
            // Variant 0 (diagonal 5-10, i.e., node4-node9 in 0-based)
            {0, 4, 6, 7},
            {4, 1, 5, 8},
            {6, 5, 2, 9},
            {9, 7, 8, 3},
            {4, 9, 6, 7},
            {4, 8, 9, 7},
            {4, 9, 8, 5},
            {4, 6, 9, 5},
            // Variant 1 (diagonal 6-8, i.e., node5-node7 in 0-based)
            {0, 4, 6, 7},
            {4, 1, 5, 8},
            {6, 5, 2, 9},
            {9, 7, 8, 3},
            {5, 6, 7, 9},
            {5, 7, 8, 9},
            {5, 8, 7, 4},
            {5, 7, 6, 4},
            // Variant 2 (diagonal 7-9, i.e., node6-node8 in 0-based)
            {0, 4, 6, 7},
            {4, 1, 5, 8},
            {6, 5, 2, 9},
            {9, 7, 8, 3},
            {6, 7, 8, 9},
            {6, 8, 5, 9},
            {6, 8, 7, 4},
            {6, 5, 8, 4}}};

        // VTK
        static constexpr int vtkCellType = 24;
        static constexpr std::array<int, 10> vtkNodeOrder = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    };

    template <>
    struct ElementTraits<Hex8>
    {
        DNDS_ELEMENT_TRAITS_COMMON(Hex8, 3, 1, 8, 8, 6, HexSpace, 8.0)

        static constexpr std::array<t_real, 3 * 8> standardCoords = {
            -1, -1, -1,
             1, -1, -1,
             1,  1, -1,
            -1,  1, -1,
            -1, -1,  1,
             1, -1,  1,
             1,  1,  1,
            -1,  1,  1};

        static constexpr ElemType GetFaceType(t_index /*iFace*/) { return Quad4; }

        static constexpr std::array<std::array<t_index, 10>, 6> faceNodes = {{
            {0, 3, 2, 1},
            {0, 1, 5, 4},
            {1, 2, 6, 5},
            {2, 3, 7, 6},
            {0, 4, 7, 3},
            {4, 5, 6, 7}}};

        static constexpr ElemType elevatedType = Hex27;
        static constexpr int numElevNodes = 19;
        // 12 edge midpoints (Line2 span), 6 face centers (Quad4 span), 1 body center (Hex8 span)
        static constexpr std::array<tElevSpan, 19> elevSpans = {{
            {0, 1}, {1, 2}, {2, 3}, {3, 0},           // bottom edges
            {0, 4}, {1, 5}, {2, 6}, {3, 7},           // vertical edges
            {4, 5}, {5, 6}, {6, 7}, {7, 4},           // top edges
            {0, 3, 2, 1}, {0, 1, 5, 4},               // face centers
            {1, 2, 6, 5}, {2, 3, 7, 6},
            {0, 4, 7, 3}, {4, 5, 6, 7},
            {0, 1, 2, 3, 4, 5, 6, 7}}};               // body center
        static constexpr std::array<ElemType, 19> elevNodeSpanTypes = {
            Line2, Line2, Line2, Line2,
            Line2, Line2, Line2, Line2,
            Line2, Line2, Line2, Line2,
            Quad4, Quad4, Quad4, Quad4, Quad4, Quad4,
            Hex8};

        // VTK
        static constexpr int vtkCellType = 12;
        static constexpr std::array<int, 8> vtkNodeOrder = {0, 1, 2, 3, 4, 5, 6, 7};
    };

    template <>
    struct ElementTraits<Hex27>
    {
        DNDS_ELEMENT_TRAITS_COMMON(Hex27, 3, 2, 8, 27, 6, HexSpace, 8.0)

        static constexpr std::array<t_real, 3 * 27> standardCoords = {
            -1, -1, -1,   1, -1, -1,   1,  1, -1,  -1,  1, -1,
            -1, -1,  1,   1, -1,  1,   1,  1,  1,  -1,  1,  1,
             0, -1, -1,   1,  0, -1,   0,  1, -1,  -1,  0, -1,
            -1, -1,  0,   1, -1,  0,   1,  1,  0,  -1,  1,  0,
             0, -1,  1,   1,  0,  1,   0,  1,  1,  -1,  0,  1,
             0,  0, -1,   0, -1,  0,   1,  0,  0,   0,  1,  0,
            -1,  0,  0,   0,  0,  1,   0,  0,  0};

        static constexpr ElemType GetFaceType(t_index /*iFace*/) { return Quad9; }

        static constexpr std::array<std::array<t_index, 10>, 6> faceNodes = {{
            {0, 3, 2, 1, 11, 10, 9, 8, 20},
            {0, 1, 5, 4, 8, 13, 16, 12, 21},
            {1, 2, 6, 5, 9, 14, 17, 13, 22},
            {2, 3, 7, 6, 10, 15, 18, 14, 23},
            {0, 4, 7, 3, 12, 19, 15, 11, 24},
            {4, 5, 6, 7, 16, 17, 18, 19, 25}}};

        static constexpr ElemType elevatedType = UnknownElem;
        static constexpr int numElevNodes = 0;

        // O2 bisect: 8 sub-hexes, 1 variant
        static constexpr int numBisect = 8;
        static constexpr int numBisectVariants = 1;
        static constexpr ElemType GetBisectElemType(t_index /*i*/) { return Hex8; }
        static constexpr std::array<tBisectSub, 8> bisectElements = {{
            {0, 8, 20, 11, 12, 21, 26, 24},
            {8, 1, 9, 20, 21, 13, 22, 26},
            {11, 20, 10, 3, 24, 26, 23, 15},
            {20, 9, 2, 10, 26, 22, 14, 23},
            {12, 21, 26, 24, 4, 16, 25, 19},
            {21, 13, 22, 26, 16, 5, 17, 25},
            {24, 26, 23, 15, 19, 25, 18, 7},
            {26, 22, 14, 23, 25, 17, 6, 18}}};

        // VTK: Hex27 has non-trivial node reordering (face-center nodes move)
        static constexpr int vtkCellType = 25;
        static constexpr std::array<int, 20> vtkNodeOrder = {
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11,
            16, 17, 18, 19,
            12, 13, 14, 15};
    };

    template <>
    struct ElementTraits<Prism6>
    {
        DNDS_ELEMENT_TRAITS_COMMON(Prism6, 3, 1, 6, 6, 5, PrismSpace, 1.0)

        static constexpr std::array<t_real, 3 * 6> standardCoords = {
            0, 0, -1,
            1, 0, -1,
            0, 1, -1,
            0, 0,  1,
            1, 0,  1,
            0, 1,  1};

        static constexpr ElemType GetFaceType(t_index iFace)
        {
            return iFace < 3 ? Quad4 : Tri3;
        }

        static constexpr std::array<std::array<t_index, 10>, 5> faceNodes = {{
            {0, 1, 4, 3},
            {1, 2, 5, 4},
            {2, 0, 3, 5},
            {0, 2, 1},
            {3, 4, 5}}};

        static constexpr ElemType elevatedType = Prism18;
        static constexpr int numElevNodes = 12;
        // 9 edge midpoints (Line2 span), 3 quad-face centers (Quad4 span)
        static constexpr std::array<tElevSpan, 12> elevSpans = {{
            {0, 1}, {1, 2}, {2, 0},                   // bottom tri edges
            {0, 3}, {1, 4}, {2, 5},                   // vertical edges
            {3, 4}, {4, 5}, {5, 3},                   // top tri edges
            {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}}}; // quad faces
        static constexpr std::array<ElemType, 12> elevNodeSpanTypes = {
            Line2, Line2, Line2,
            Line2, Line2, Line2,
            Line2, Line2, Line2,
            Quad4, Quad4, Quad4};

        // VTK
        static constexpr int vtkCellType = 13;
        static constexpr std::array<int, 6> vtkNodeOrder = {0, 1, 2, 3, 4, 5};
    };

    template <>
    struct ElementTraits<Prism18>
    {
        DNDS_ELEMENT_TRAITS_COMMON(Prism18, 3, 2, 6, 18, 5, PrismSpace, 1.0)

        static constexpr std::array<t_real, 3 * 18> standardCoords = {
            0, 0, -1,   1, 0, -1,   0, 1, -1,
            0, 0,  1,   1, 0,  1,   0, 1,  1,
            0.5, 0, -1,  0.5, 0.5, -1,  0, 0.5, -1,
            0, 0, 0,     1, 0, 0,    0, 1, 0,
            0.5, 0, 1,   0.5, 0.5, 1,  0, 0.5, 1,
            0.5, 0, 0,   0.5, 0.5, 0,  0, 0.5, 0};

        static constexpr ElemType GetFaceType(t_index iFace)
        {
            return iFace < 3 ? Quad9 : Tri6;
        }

        static constexpr std::array<std::array<t_index, 10>, 5> faceNodes = {{
            {0, 1, 4, 3, 6, 10, 12, 9, 15},
            {1, 2, 5, 4, 7, 11, 13, 10, 16},
            {2, 0, 3, 5, 8, 9, 14, 11, 17},
            {0, 2, 1, 8, 7, 6},
            {3, 4, 5, 12, 13, 14}}};

        static constexpr ElemType elevatedType = UnknownElem;
        static constexpr int numElevNodes = 0;

        // O2 bisect: 8 sub-prisms, 1 variant
        static constexpr int numBisect = 8;
        static constexpr int numBisectVariants = 1;
        static constexpr ElemType GetBisectElemType(t_index /*i*/) { return Prism6; }
        static constexpr std::array<tBisectSub, 8> bisectElements = {{
            {0, 6, 8, 9, 15, 17},
            {6, 1, 7, 15, 10, 16},
            {8, 6, 7, 17, 15, 16},
            {8, 7, 2, 17, 16, 11},
            {9, 15, 17, 3, 12, 14},
            {15, 10, 16, 12, 4, 13},
            {17, 15, 16, 14, 12, 13},
            {17, 16, 11, 14, 13, 5}}};

        // VTK: Prism18 has non-trivial node reordering
        static constexpr int vtkCellType = 26;
        static constexpr std::array<int, 15> vtkNodeOrder = {
            0, 1, 2, 3, 4, 5,
            6, 7, 8,
            12, 13, 14,
            9, 10, 11};
    };

    template <>
    struct ElementTraits<Pyramid5>
    {
        DNDS_ELEMENT_TRAITS_COMMON(Pyramid5, 3, 1, 5, 5, 5, PyramidSpace, 4.0 / 3.0)

        static constexpr std::array<t_real, 3 * 5> standardCoords = {
            -1, -1, 0,
             1, -1, 0,
             1,  1, 0,
            -1,  1, 0,
             0,  0, 1};

        static constexpr ElemType GetFaceType(t_index iFace)
        {
            return iFace < 1 ? Quad4 : Tri3;
        }

        static constexpr std::array<std::array<t_index, 10>, 5> faceNodes = {{
            {0, 3, 2, 1},
            {0, 1, 4},
            {1, 2, 4},
            {2, 3, 4},
            {3, 0, 4}}};

        static constexpr ElemType elevatedType = Pyramid14;
        static constexpr int numElevNodes = 9;
        // 8 edge midpoints (Line2 span), 1 quad-base center (Quad4 span)
        static constexpr std::array<tElevSpan, 9> elevSpans = {{
            {0, 1}, {1, 2}, {2, 3}, {3, 0},           // base edges
            {0, 4}, {1, 4}, {2, 4}, {3, 4},           // lateral edges
            {0, 3, 2, 1}}};                            // base face center
        static constexpr std::array<ElemType, 9> elevNodeSpanTypes = {
            Line2, Line2, Line2, Line2,
            Line2, Line2, Line2, Line2,
            Quad4};

        // VTK
        static constexpr int vtkCellType = 14;
        static constexpr std::array<int, 5> vtkNodeOrder = {0, 1, 2, 3, 4};
    };

    template <>
    struct ElementTraits<Pyramid14>
    {
        DNDS_ELEMENT_TRAITS_COMMON(Pyramid14, 3, 2, 5, 14, 5, PyramidSpace, 4.0 / 3.0)

        static constexpr std::array<t_real, 3 * 14> standardCoords = {
            -1, -1, 0,    1, -1, 0,    1,  1, 0,   -1,  1, 0,    0,  0, 1,
             0, -1, 0,    1,  0, 0,    0,  1, 0,   -1,  0, 0,
            -0.5, -0.5, 0.5,  0.5, -0.5, 0.5,  0.5, 0.5, 0.5,  -0.5, 0.5, 0.5,
             0, 0, 0};

        static constexpr ElemType GetFaceType(t_index iFace)
        {
            return iFace < 1 ? Quad9 : Tri6;
        }

        static constexpr std::array<std::array<t_index, 10>, 5> faceNodes = {{
            {0, 3, 2, 1, 8, 7, 6, 5, 13},
            {0, 1, 4, 5, 10, 9},
            {1, 2, 4, 6, 11, 10},
            {2, 3, 4, 7, 12, 11},
            {3, 0, 4, 8, 9, 12}}};

        static constexpr ElemType elevatedType = UnknownElem;
        static constexpr int numElevNodes = 0;

        // O2 bisect: 12 sub-elements per variant, 2 variants
        // First 4 sub-elements are Pyramid5, remaining 8 are Tet4
        static constexpr int numBisect = 12;
        static constexpr int numBisectVariants = 2;
        static constexpr ElemType GetBisectElemType(t_index i)
        {
            return i < 4 ? Pyramid5 : Tet4;
        }
        // Variant 0 (12), variant 1 (12) = 24 rows
        static constexpr std::array<tBisectSub, 24> bisectElements = {{
            // Variant 0
            {0, 5, 13, 8, 9},
            {5, 1, 6, 13, 10},
            {8, 13, 7, 3, 12},
            {13, 6, 2, 7, 11},
            {12, 9, 8, 13},
            {9, 10, 5, 13},
            {10, 11, 6, 13},
            {11, 12, 7, 13},
            {9, 11, 12, 4},
            {9, 10, 11, 4},
            {9, 11, 10, 13},
            {9, 12, 11, 13},
            // Variant 1
            {0, 5, 13, 8, 9},
            {5, 1, 6, 13, 10},
            {8, 13, 7, 3, 12},
            {13, 6, 2, 7, 11},
            {12, 9, 8, 13},
            {9, 10, 5, 13},
            {10, 11, 6, 13},
            {11, 12, 7, 13},
            {10, 12, 9, 4},
            {10, 11, 12, 4},
            {10, 12, 11, 13},
            {10, 9, 12, 13}}};

        // VTK
        static constexpr int vtkCellType = 27;
        static constexpr std::array<int, 13> vtkNodeOrder = {
            0, 1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12};
    };

#undef DNDS_ELEMENT_TRAITS_COMMON

    // ----------------------------------------------------------------
    // ParamSpace-level information (not per-element, per-space)
    // ----------------------------------------------------------------
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
