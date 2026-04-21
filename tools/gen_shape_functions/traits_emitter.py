"""
C++ code emitter for ElementTraits topology data.

Generates ElementTraits<ElemType> specializations from the authoritative
element_data.py. Output includes:
  - standardCoords
  - faceNodes, GetFaceType, numFaces
  - edgeNodes, GetEdgeType, numEdges (3D only)
  - elevSpans, elevNodeSpanTypes, elevatedType, numElevNodes (O1 only)
  - bisectElements, GetBisectElemType, numBisect, numBisectVariants (O2 only)
  - vtkCellType, vtkNodeOrder

The generated code is placed between GEN_ELEM_TRAITS_BEGIN/END markers.
"""

import sys
import os

# Add project root to path so we can import element_data
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from tools.elements.element_data import ElementData, BY_DNDSR_NAME


def _fmt_coord(c):
    """Format a coordinate value, using int when possible."""
    if c == int(c):
        return str(int(c))
    # Use exact fractions for common values
    if abs(c - 0.5) < 1e-15:
        return "0.5"
    if abs(c + 0.5) < 1e-15:
        return "-0.5"
    return str(c)


def _fmt_coord_triple(x, y, z):
    """Format a (x, y, z) coordinate as C++ initializer."""
    return f"{_fmt_coord(x)}, {_fmt_coord(y)}, {_fmt_coord(z)}"


def emit_elem_traits(elem_name):
    """Generate C++ lines for ElementTraits<ElemType> specialization.

    Args:
        elem_name: DNDSR element name (e.g. "Tet4")

    Returns:
        List of C++ source lines (no leading newline, with proper indentation).
    """
    if elem_name not in BY_DNDSR_NAME:
        raise ValueError(f"Unknown element: {elem_name}")

    e = BY_DNDSR_NAME[elem_name]
    I = "    "     # 4-space indent (inside namespace)
    I2 = "        " # 8-space indent (inside struct)

    lines = []

    # --- Struct open ---
    lines.append(f"")
    lines.append(f"{I}template <>")
    lines.append(f"{I}struct ElementTraits<{e.dndsr_name}>")
    lines.append(f"{I}{{")

    # --- Core identification ---
    lines.append(f"{I2}static constexpr ElemType elemType = {e.dndsr_name};")
    lines.append(f"{I2}static constexpr int dim = {e.dim};")
    lines.append(f"{I2}static constexpr int order = {e.order};")
    lines.append(f"{I2}static constexpr int numVertices = {e.num_vertices};")
    lines.append(f"{I2}static constexpr int numNodes = {e.num_nodes};")
    lines.append(f"{I2}static constexpr int numFaces = {e.num_faces};")
    if e.num_edges > 0:
        lines.append(f"{I2}static constexpr int numEdges = {e.num_edges};")
    lines.append(f"{I2}static constexpr ParamSpace paramSpace = {e.param_space};")
    lines.append(f"{I2}static constexpr t_real paramSpaceVol = {e.param_space_vol};")
    lines.append(f"")

    # --- Standard coordinates ---
    coords_0 = e.node_coords_0based()
    lines.append(f"{I2}static constexpr std::array<t_real, 3 * {e.num_nodes}> standardCoords = {{")
    for i, (x, y, z) in enumerate(coords_0):
        comma = "," if i < e.num_nodes - 1 else "};"
        comment = ""
        if i < e.num_vertices:
            comment = f"  // Node {i}: vertex"
        else:
            comment = f"  // Node {i}"
        lines.append(f"{I2}    {_fmt_coord_triple(x, y, z)}{comma}{comment}")
    lines.append(f"")

    # --- Face topology ---
    # GetFaceType must always be present (dispatch calls it for all element types)
    if e.num_faces == 0:
        lines.append(f"{I2}static constexpr ElemType GetFaceType(t_index /*iFace*/) {{ return UnknownElem; }}")
    else:
        unique_types = list(set(e.face_types))
        if len(unique_types) == 1:
            lines.append(f"{I2}static constexpr ElemType GetFaceType(t_index /*iFace*/) {{ return {unique_types[0]}; }}")
        else:
            lines.append(f"{I2}static constexpr ElemType GetFaceType(t_index iFace)")
            lines.append(f"{I2}{{")
            # Build switch or conditional chain
            for fi, ft in enumerate(e.face_types):
                if fi == 0:
                    lines.append(f"{I2}    if (iFace < {_find_type_boundary(e.face_types, fi)}) return {ft};")
                elif ft != e.face_types[fi - 1]:
                    lines.append(f"{I2}    return {ft};")
                    break
            else:
                lines.append(f"{I2}    return {e.face_types[-1]};")
            lines.append(f"{I2}}}")

        # faceNodes
        face_nodes_0 = e.face_nodes_0based()
        max_face_nodes = max(len(f) for f in face_nodes_0)
        pad_width = 10  # existing convention
        lines.append(f"")
        lines.append(f"{I2}static constexpr std::array<std::array<t_index, {pad_width}>, {e.num_faces}> faceNodes = {{{{")
        for fi, fn in enumerate(face_nodes_0):
            vals = ", ".join(str(n) for n in fn)
            comma = "," if fi < e.num_faces - 1 else ""
            lines.append(f"{I2}    {{{vals}}}{comma}")
        lines.append(f"{I2}}}}};")

    # --- Edge topology (3D only) ---
    if e.num_edges > 0:
        lines.append(f"")
        edge_nodes_0 = e.edge_nodes_0based()
        edge_width = max(len(edge) for edge in edge_nodes_0)
        pad_edge = max(edge_width, 3)  # Line3 has 3 nodes

        # GetEdgeType
        lines.append(f"{I2}static constexpr ElemType GetEdgeType(t_index /*iEdge*/) {{ return {e.edge_type}; }}")
        lines.append(f"")

        lines.append(f"{I2}static constexpr std::array<std::array<t_index, {pad_edge}>, {e.num_edges}> edgeNodes = {{{{")
        for ei, en in enumerate(edge_nodes_0):
            vals = ", ".join(str(n) for n in en)
            comma = "," if ei < e.num_edges - 1 else ""
            lines.append(f"{I2}    {{{vals}}}{comma}")
        lines.append(f"{I2}}}}};")

    # --- Elevation (O1 -> O2) ---
    lines.append(f"")
    lines.append(f"{I2}static constexpr ElemType elevatedType = {e.elevated_type};")
    lines.append(f"{I2}static constexpr int numElevNodes = {e.num_elev_nodes};")

    if e.num_elev_nodes > 0:
        elev_0 = e.elev_spans_0based()
        lines.append(f"")
        lines.append(f"{I2}static constexpr std::array<tElevSpan, {e.num_elev_nodes}> elevSpans = {{{{")
        for si, span in enumerate(elev_0):
            vals = ", ".join(str(n) for n in span)
            comma = "," if si < e.num_elev_nodes - 1 else ""
            lines.append(f"{I2}    {{{vals}}}{comma}")
        lines.append(f"{I2}}}}};")

        lines.append(f"")
        types_str = ", ".join(e.elev_span_types)
        lines.append(f"{I2}static constexpr std::array<ElemType, {e.num_elev_nodes}> elevNodeSpanTypes = {{")
        lines.append(f"{I2}    {types_str}}};")

    # --- Bisection (O2 -> O1) ---
    if e.num_bisect > 0:
        lines.append(f"")
        lines.append(f"{I2}static constexpr int numBisect = {e.num_bisect};")
        lines.append(f"{I2}static constexpr int numBisectVariants = {e.num_bisect_variants};")

        # GetBisectElemType
        lines.append(f"")
        if e.bisect_elem_types is None:
            # Uniform type
            lines.append(f"{I2}static constexpr ElemType GetBisectElemType(t_index /*i*/) {{ return {e.bisect_uniform_type}; }}")
        else:
            # Non-uniform (e.g. Pyramid14: first 4 are Pyramid5, rest are Tet4)
            lines.append(f"{I2}static constexpr ElemType GetBisectElemType(t_index i)")
            lines.append(f"{I2}{{")
            # Find boundaries between types
            prev_type = e.bisect_elem_types[0]
            boundary = 0
            for bi, bt in enumerate(e.bisect_elem_types):
                if bt != prev_type:
                    lines.append(f"{I2}    if (i < {bi}) return {prev_type};")
                    prev_type = bt
                    boundary = bi
            lines.append(f"{I2}    return {prev_type};")
            lines.append(f"{I2}}}")

        # bisectElements
        bisect_0 = e.bisect_elements_0based()
        total = len(bisect_0)
        lines.append(f"")
        lines.append(f"{I2}static constexpr std::array<tBisectSub, {total}> bisectElements = {{{{")
        subs_per_variant = e.num_bisect
        for bi, sub in enumerate(bisect_0):
            vals = ", ".join(str(n) for n in sub)
            comma = "," if bi < total - 1 else ""
            # Add variant comment
            variant_idx = bi // subs_per_variant
            sub_idx = bi % subs_per_variant
            if sub_idx == 0 and e.num_bisect_variants > 1:
                lines.append(f"{I2}    // Variant {variant_idx}")
            lines.append(f"{I2}    {{{vals}}}{comma}")
        lines.append(f"{I2}}}}};")

    # --- VTK ---
    lines.append(f"")
    lines.append(f"{I2}static constexpr int vtkCellType = {e.vtk_cell_type};")
    vtk_str = ", ".join(str(v) for v in e.vtk_node_order)
    lines.append(f"")
    lines.append(f"{I2}static constexpr std::array<int, {len(e.vtk_node_order)}> vtkNodeOrder = {{{vtk_str}}};")

    # --- Struct close ---
    lines.append(f"{I}}};")

    return lines


def _find_type_boundary(face_types, start_idx):
    """Find the first index where face type differs from face_types[start_idx]."""
    t = face_types[start_idx]
    for i in range(start_idx, len(face_types)):
        if face_types[i] != t:
            return i
    return len(face_types)
