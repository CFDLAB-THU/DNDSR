#pragma once
// ElementTraitsBase.hpp -- Shared type definitions for element traits
//
// This header provides the common type aliases used by ElementTraits
// specializations. It is included by both ElementTraits.hpp and the
// individual element trait files in Elements/.

#include <array>
#include <cstdint>
#include "DNDS/Defines.hpp"
#include "Geometric.hpp"
#include "ElemEnum.hpp"

namespace DNDS::Geom::Elem
{

    // ----------------------------------------------------------------
    // ElementTraits<ElemType t> -- primary template (unspecialized = error)
    // Must be specialized for each valid ElemType in Elements/*.hpp files
    // ----------------------------------------------------------------
    template <ElemType t>
    struct ElementTraits; // Intentionally undefined - forces specialization

    // ----------------------------------------------------------------
    // Macro to reduce boilerplate in each specialization
    // ----------------------------------------------------------------
    /**
     * @brief Common element trait definitions
     * @param ETYPE    Element type enum value (e.g., Line2, Tri3)
     * @param DIM_     Spatial dimension (1, 2, or 3)
     * @param ORDER_   Polynomial order of shape functions
     * @param NV_      Number of vertices (corner nodes)
     * @param NN_      Total number of nodes (including edge/face/internal)
     * @param NF_      Number of faces
     * @param PSPACE_  Parametric space type (LineSpace, TriSpace, etc.)
     * @param PSVOL_   Volume of parametric space (length, area, or volume)
     */
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
    // Type Aliases for Element Trait Arrays
    // ----------------------------------------------------------------

    /**
     * @brief Maximum width of elevation span (Hex8 body center uses 8 nodes)
     * 
     * Elevation spans define which parent nodes are connected to create
     * new nodes during p-refinement (order elevation). For example:
     * - Edge midpoint: span of 2 nodes
     * - Quad face center: span of 4 nodes  
     * - Hex body center: span of 8 nodes
     */
    static constexpr int elevSpanMaxWidth = 8;
    
    /**
     * @brief Elevation span type: array of parent node indices
     * 
     * Used to define which existing nodes define the position of a new
     * node created during element order elevation.
     */
    using tElevSpan = std::array<t_index, elevSpanMaxWidth>;

    /**
     * @brief Maximum nodes in a bisection sub-element (Hex8 = 8 nodes)
     * 
     * Bisection (h-refinement) splits an element into smaller sub-elements.
     * This defines the maximum nodes any sub-element can have.
     */
    static constexpr int bisectSubMaxNodes = 8;
    
    /**
     * @brief Bisection sub-element node array type
     * 
     * Maps local node indices of a sub-element to parent element nodes.
     * Used in adaptive mesh refinement to define child element connectivity.
     */
    using tBisectSub = std::array<t_index, bisectSubMaxNodes>;

    /**
     * @brief Maximum nodes in an edge (Line3 = 3 nodes for quadratic)
     *
     * Edge sub-entities of 3D elements. Linear elements have 2-node edges
     * (Line2), quadratic elements have 3-node edges (Line3).
     */
    static constexpr int edgeNodeMaxWidth = 3;

    /**
     * @brief Edge node array type: local node indices forming one edge.
     *
     * For Line2 edges, only the first 2 entries are valid.
     * For Line3 edges, all 3 entries are valid.
     */
    using tEdgeNodes = std::array<t_index, edgeNodeMaxWidth>;

    /**
     * @brief Maximum VTK nodes (Hex27 = 27 nodes)
     * 
     * VTK uses different node ordering than internal DNDS representation
     * for higher-order elements. This defines the maximum array size needed.
     */
    static constexpr int vtkNodeOrderMax = 27;
    
    /**
     * @brief VTK node ordering permutation array type
     * 
     * Maps DNDS node indices to VTK node indices:
     *   vtkNodeOrder[i] = VTK index of DNDS node i
     * 
     * Only first numNodes entries are valid; rest are padding.
     */
    using tVTKNodeOrder = std::array<int, vtkNodeOrderMax>;

} // namespace DNDS::Geom::Elem
