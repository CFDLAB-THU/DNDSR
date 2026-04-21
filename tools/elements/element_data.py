"""
Authoritative element topology data following CGNS SIDS convention.

Source of truth: https://cgns.org/standard/SIDS/convention.html
Transcribed data: tools/elements/cgns_topology_1based.txt

All node indices in this module are 1-based (CGNS convention).
The generator converts to 0-based when emitting C++ code.

Node coordinate data uses the CGNS reference parametric spaces:
  - Line:     xi in [-1, 1]
  - Triangle: (xi, et) in {xi>=0, et>=0, xi+et<=1}
  - Quad:     (xi, et) in [-1,1] x [-1,1]
  - Tet:      (xi, et, zt) in {xi>=0, et>=0, zt>=0, xi+et+zt<=1}
  - Hex:      (xi, et, zt) in [-1,1]^3
  - Prism:    tri(xi,et) x zt in [-1,1]
  - Pyramid:  quad(xi,et) at zt=0, apex at zt=1
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from fractions import Fraction

# Type aliases for readability (all 1-based CGNS indices)
NodeCoord = Tuple[float, float, float]  # (x, y, z)
Edge1 = Tuple[int, ...]                 # (N_a, N_b) or (N_a, N_b, N_mid)
Face1 = List[int]                       # [N_a, N_b, N_c, ...] with optional mid-edge/mid-face nodes
ElevSpan1 = Tuple[int, ...]             # parent node indices defining a new elevation node
BisectSub1 = Tuple[int, ...]            # sub-element local node indices


@dataclass
class ElementData:
    """Complete topology for one element type (CGNS convention, 1-based)."""

    # --- Identity ---
    cgns_name: str                       # e.g. "TETRA_4"
    dndsr_name: str                      # e.g. "Tet4"  (C++ ElemType enum)
    dim: int                             # parametric dimension (1, 2, or 3)
    order: int                           # polynomial order (1 or 2)
    num_vertices: int                    # corner node count
    num_nodes: int                       # total node count
    param_space: str                     # C++ ParamSpace enum name
    param_space_vol: str                 # C++ expression for reference domain volume

    # --- Geometry (1-based node indices -> coordinates) ---
    node_coords: Dict[int, NodeCoord]    # {1: (x,y,z), 2: (x,y,z), ...}

    # --- Face topology (1-based) ---
    num_faces: int
    face_types: List[str]                # per-face C++ ElemType name
    face_nodes: List[Face1]              # per-face node lists (1-based)

    # --- Edge topology (1-based, 3D only) ---
    num_edges: int = 0
    edge_type: str = "UnknownElem"       # C++ ElemType for edges (Line2/Line3)
    edge_nodes: List[Edge1] = field(default_factory=list)
    # Visual grouping for diagrams: list of (label, [edge_indices_0based])
    # e.g. [("bottom", [0,1,2,3]), ("lateral", [4,5]), ("lateral", [6,7])]
    edge_groups: List[Tuple[str, List[int]]] = field(default_factory=list)

    # --- Elevation: O1 -> O2 (only on O1 elements) ---
    elevated_type: str = "UnknownElem"
    num_elev_nodes: int = 0
    elev_spans: List[ElevSpan1] = field(default_factory=list)
    elev_span_types: List[str] = field(default_factory=list)

    # --- Bisection: O2 -> O1 (only on O2 elements) ---
    num_bisect: int = 0
    num_bisect_variants: int = 0
    bisect_elem_types: Optional[List[str]] = None  # per-sub-element type; None = uniform
    bisect_uniform_type: str = "UnknownElem"       # when all sub-elements have same type
    bisect_elements: List[BisectSub1] = field(default_factory=list)

    # --- VTK ---
    vtk_cell_type: int = 0
    vtk_node_order: List[int] = field(default_factory=list)  # 0-based permutation

    def node_coords_0based(self) -> List[NodeCoord]:
        """Return coordinates in 0-based order."""
        return [self.node_coords[i + 1] for i in range(self.num_nodes)]

    def face_nodes_0based(self) -> List[List[int]]:
        """Return face_nodes converted to 0-based."""
        return [[n - 1 for n in face] for face in self.face_nodes]

    def edge_nodes_0based(self) -> List[List[int]]:
        """Return edge_nodes converted to 0-based."""
        return [[n - 1 for n in edge] for edge in self.edge_nodes]

    def elev_spans_0based(self) -> List[List[int]]:
        """Return elev_spans converted to 0-based."""
        return [[n - 1 for n in span] for span in self.elev_spans]

    def bisect_elements_0based(self) -> List[List[int]]:
        """Return bisect_elements converted to 0-based."""
        return [[n - 1 for n in sub] for sub in self.bisect_elements]


# ======================================================================
# Helper: midpoint coordinate
# ======================================================================

def _mid(*coords):
    """Average of N coordinate tuples."""
    n = len(coords)
    return tuple(sum(c[i] for c in coords) / n for i in range(3))


# ======================================================================
# 1D Elements
# ======================================================================

LINE2 = ElementData(
    cgns_name="BAR_2", dndsr_name="Line2",
    dim=1, order=1, num_vertices=2, num_nodes=2,
    param_space="LineSpace", param_space_vol="2.0",
    node_coords={1: (-1, 0, 0), 2: (1, 0, 0)},
    num_faces=0, face_types=[], face_nodes=[],
    elevated_type="Line3", num_elev_nodes=1,
    elev_spans=[(1, 2)],
    elev_span_types=["Line2"],
    vtk_cell_type=3, vtk_node_order=[0, 1],
)

LINE3 = ElementData(
    cgns_name="BAR_3", dndsr_name="Line3",
    dim=1, order=2, num_vertices=2, num_nodes=3,
    param_space="LineSpace", param_space_vol="2.0",
    node_coords={1: (-1, 0, 0), 2: (1, 0, 0), 3: (0, 0, 0)},
    num_faces=0, face_types=[], face_nodes=[],
    num_bisect=2, num_bisect_variants=1,
    bisect_uniform_type="Line2",
    bisect_elements=[(1, 3), (3, 2)],
    vtk_cell_type=4, vtk_node_order=[0, 2, 1],  # VTK: midpoint at position 1
)

# ======================================================================
# 2D Elements
# ======================================================================

TRI3 = ElementData(
    cgns_name="TRI_3", dndsr_name="Tri3",
    dim=2, order=1, num_vertices=3, num_nodes=3,
    param_space="TriSpace", param_space_vol="0.5",
    node_coords={1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0)},
    num_faces=3,
    face_types=["Line2", "Line2", "Line2"],
    face_nodes=[[1, 2], [2, 3], [3, 1]],
    elevated_type="Tri6", num_elev_nodes=3,
    elev_spans=[(1, 2), (2, 3), (3, 1)],
    elev_span_types=["Line2", "Line2", "Line2"],
    vtk_cell_type=5, vtk_node_order=[0, 1, 2],
)

TRI6 = ElementData(
    cgns_name="TRI_6", dndsr_name="Tri6",
    dim=2, order=2, num_vertices=3, num_nodes=6,
    param_space="TriSpace", param_space_vol="0.5",
    node_coords={
        1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0),
        4: (0.5, 0, 0), 5: (0.5, 0.5, 0), 6: (0, 0.5, 0),
    },
    num_faces=3,
    face_types=["Line3", "Line3", "Line3"],
    face_nodes=[[1, 2, 4], [2, 3, 5], [3, 1, 6]],
    num_bisect=4, num_bisect_variants=1,
    bisect_uniform_type="Tri3",
    bisect_elements=[(1, 4, 6), (4, 2, 5), (6, 4, 5), (6, 5, 3)],
    vtk_cell_type=22, vtk_node_order=[0, 1, 2, 3, 4, 5],
)

QUAD4 = ElementData(
    cgns_name="QUAD_4", dndsr_name="Quad4",
    dim=2, order=1, num_vertices=4, num_nodes=4,
    param_space="QuadSpace", param_space_vol="4.0",
    node_coords={
        1: (-1, -1, 0), 2: (1, -1, 0), 3: (1, 1, 0), 4: (-1, 1, 0),
    },
    num_faces=4,
    face_types=["Line2", "Line2", "Line2", "Line2"],
    face_nodes=[[1, 2], [2, 3], [3, 4], [4, 1]],
    elevated_type="Quad9", num_elev_nodes=5,
    elev_spans=[(1, 2), (2, 3), (3, 4), (4, 1), (1, 2, 3, 4)],
    elev_span_types=["Line2", "Line2", "Line2", "Line2", "Quad4"],
    vtk_cell_type=9, vtk_node_order=[0, 1, 2, 3],
)

QUAD9 = ElementData(
    cgns_name="QUAD_9", dndsr_name="Quad9",
    dim=2, order=2, num_vertices=4, num_nodes=9,
    param_space="QuadSpace", param_space_vol="4.0",
    node_coords={
        1: (-1, -1, 0), 2: (1, -1, 0), 3: (1, 1, 0), 4: (-1, 1, 0),
        5: (0, -1, 0), 6: (1, 0, 0), 7: (0, 1, 0), 8: (-1, 0, 0),
        9: (0, 0, 0),
    },
    num_faces=4,
    face_types=["Line3", "Line3", "Line3", "Line3"],
    face_nodes=[[1, 2, 5], [2, 3, 6], [3, 4, 7], [4, 1, 8]],
    num_bisect=4, num_bisect_variants=1,
    bisect_uniform_type="Quad4",
    bisect_elements=[(1, 5, 9, 8), (5, 2, 6, 9), (8, 9, 7, 4), (9, 6, 3, 7)],
    vtk_cell_type=23, vtk_node_order=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)

# ======================================================================
# 3D Elements
# ======================================================================

TET4 = ElementData(
    cgns_name="TETRA_4", dndsr_name="Tet4",
    dim=3, order=1, num_vertices=4, num_nodes=4,
    param_space="TetSpace", param_space_vol="1.0 / 6.0",
    node_coords={
        1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0), 4: (0, 0, 1),
    },
    num_faces=4,
    face_types=["Tri3", "Tri3", "Tri3", "Tri3"],
    face_nodes=[[1, 3, 2], [1, 2, 4], [2, 3, 4], [3, 1, 4]],
    num_edges=6, edge_type="Line2",
    edge_nodes=[(1, 2), (2, 3), (3, 1), (1, 4), (2, 4), (3, 4)],
    edge_groups=[("base", [0, 1, 2]), ("lateral", [3, 4, 5])],
    elevated_type="Tet10", num_elev_nodes=6,
    elev_spans=[(1, 2), (2, 3), (3, 1), (1, 4), (2, 4), (3, 4)],
    elev_span_types=["Line2"] * 6,
    vtk_cell_type=10, vtk_node_order=[0, 1, 2, 3],
)

TET10 = ElementData(
    cgns_name="TETRA_10", dndsr_name="Tet10",
    dim=3, order=2, num_vertices=4, num_nodes=10,
    param_space="TetSpace", param_space_vol="1.0 / 6.0",
    node_coords={
        1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0), 4: (0, 0, 1),
        5: (0.5, 0, 0), 6: (0.5, 0.5, 0), 7: (0, 0.5, 0),
        8: (0, 0, 0.5), 9: (0.5, 0, 0.5), 10: (0, 0.5, 0.5),
    },
    num_faces=4,
    face_types=["Tri6", "Tri6", "Tri6", "Tri6"],
    face_nodes=[
        [1, 3, 2, 7, 6, 5],
        [1, 2, 4, 5, 9, 8],
        [2, 3, 4, 6, 10, 9],
        [3, 1, 4, 7, 8, 10],
    ],
    num_edges=6, edge_type="Line3",
    edge_nodes=[
        (1, 2, 5), (2, 3, 6), (3, 1, 7),
        (1, 4, 8), (2, 4, 9), (3, 4, 10),
    ],
    edge_groups=[("base", [0, 1, 2]), ("lateral", [3, 4, 5])],
    num_bisect=8, num_bisect_variants=3,
    bisect_uniform_type="Tet4",
    bisect_elements=[
        # Variant 0 (diagonal 5-10)
        (1, 5, 7, 8), (5, 2, 6, 9), (7, 6, 3, 10), (10, 8, 9, 4),
        (5, 10, 7, 8), (5, 9, 10, 8), (5, 10, 9, 6), (5, 7, 10, 6),
        # Variant 1 (diagonal 6-8)
        (1, 5, 7, 8), (5, 2, 6, 9), (7, 6, 3, 10), (10, 8, 9, 4),
        (6, 7, 8, 10), (6, 8, 9, 10), (6, 9, 8, 5), (6, 8, 7, 5),
        # Variant 2 (diagonal 7-9)
        (1, 5, 7, 8), (5, 2, 6, 9), (7, 6, 3, 10), (10, 8, 9, 4),
        (7, 8, 9, 10), (7, 9, 6, 10), (7, 9, 8, 5), (7, 6, 9, 5),
    ],
    vtk_cell_type=24, vtk_node_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
)

PYRAMID5 = ElementData(
    cgns_name="PYRA_5", dndsr_name="Pyramid5",
    dim=3, order=1, num_vertices=5, num_nodes=5,
    param_space="PyramidSpace", param_space_vol="4.0 / 3.0",
    node_coords={
        1: (-1, -1, 0), 2: (1, -1, 0), 3: (1, 1, 0), 4: (-1, 1, 0),
        5: (0, 0, 1),
    },
    num_faces=5,
    face_types=["Quad4", "Tri3", "Tri3", "Tri3", "Tri3"],
    face_nodes=[
        [1, 4, 3, 2],  # F1: base quad
        [1, 2, 5],      # F2: side tri
        [2, 3, 5],      # F3: side tri
        [3, 4, 5],      # F4: side tri
        [4, 1, 5],      # F5: side tri
    ],
    num_edges=8, edge_type="Line2",
    edge_nodes=[
        (1, 2), (2, 3), (3, 4), (4, 1),  # base
        (1, 5), (2, 5), (3, 5), (4, 5),  # lateral
    ],
    edge_groups=[("base", [0, 1, 2, 3]), ("lateral", [4, 5]), ("lateral", [6, 7])],
    elevated_type="Pyramid14", num_elev_nodes=9,
    elev_spans=[
        (1, 2), (2, 3), (3, 4), (4, 1),  # base edges
        (1, 5), (2, 5), (3, 5), (4, 5),  # lateral edges
        (1, 4, 3, 2),                      # base face center
    ],
    elev_span_types=["Line2"] * 8 + ["Quad4"],
    vtk_cell_type=14, vtk_node_order=[0, 1, 2, 3, 4],
)

PYRAMID14 = ElementData(
    cgns_name="PYRA_14", dndsr_name="Pyramid14",
    dim=3, order=2, num_vertices=5, num_nodes=14,
    param_space="PyramidSpace", param_space_vol="4.0 / 3.0",
    node_coords={
        1: (-1, -1, 0), 2: (1, -1, 0), 3: (1, 1, 0), 4: (-1, 1, 0),
        5: (0, 0, 1),
        6: (0, -1, 0), 7: (1, 0, 0), 8: (0, 1, 0), 9: (-1, 0, 0),  # base edge mids
        10: (-0.5, -0.5, 0.5), 11: (0.5, -0.5, 0.5),                # lateral edge mids
        12: (0.5, 0.5, 0.5), 13: (-0.5, 0.5, 0.5),
        14: (0, 0, 0),  # base face center
    },
    num_faces=5,
    face_types=["Quad9", "Tri6", "Tri6", "Tri6", "Tri6"],
    face_nodes=[
        [1, 4, 3, 2, 9, 8, 7, 6, 14],  # F1: base quad9
        [1, 2, 5, 6, 11, 10],           # F2: side tri6
        [2, 3, 5, 7, 12, 11],           # F3: side tri6
        [3, 4, 5, 8, 13, 12],           # F4: side tri6
        [4, 1, 5, 9, 10, 13],           # F5: side tri6
    ],
    num_edges=8, edge_type="Line3",
    edge_nodes=[
        (1, 2, 6), (2, 3, 7), (3, 4, 8), (4, 1, 9),    # base
        (1, 5, 10), (2, 5, 11), (3, 5, 12), (4, 5, 13),  # lateral
    ],
    edge_groups=[("base", [0, 1, 2, 3]), ("lateral", [4, 5]), ("lateral", [6, 7])],
    num_bisect=12, num_bisect_variants=2,
    bisect_elem_types=["Pyramid5"] * 4 + ["Tet4"] * 8,
    bisect_elements=[
        # Variant 0
        (1, 6, 14, 9, 10), (6, 2, 7, 14, 11),
        (9, 14, 8, 4, 13), (14, 7, 3, 8, 12),
        (13, 10, 9, 14), (10, 11, 6, 14),
        (11, 12, 7, 14), (12, 13, 8, 14),
        (10, 12, 13, 5), (10, 11, 12, 5),
        (10, 12, 11, 14), (10, 13, 12, 14),
        # Variant 1
        (1, 6, 14, 9, 10), (6, 2, 7, 14, 11),
        (9, 14, 8, 4, 13), (14, 7, 3, 8, 12),
        (13, 10, 9, 14), (10, 11, 6, 14),
        (11, 12, 7, 14), (12, 13, 8, 14),
        (11, 13, 10, 5), (11, 12, 13, 5),
        (11, 13, 12, 14), (11, 10, 13, 14),
    ],
    vtk_cell_type=27,
    vtk_node_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
)

PRISM6 = ElementData(
    cgns_name="PENTA_6", dndsr_name="Prism6",
    dim=3, order=1, num_vertices=6, num_nodes=6,
    param_space="PrismSpace", param_space_vol="1.0",
    node_coords={
        1: (0, 0, -1), 2: (1, 0, -1), 3: (0, 1, -1),
        4: (0, 0, 1), 5: (1, 0, 1), 6: (0, 1, 1),
    },
    num_faces=5,
    face_types=["Quad4", "Quad4", "Quad4", "Tri3", "Tri3"],
    face_nodes=[
        [1, 2, 5, 4],  # F1: side quad
        [2, 3, 6, 5],  # F2: side quad
        [3, 1, 4, 6],  # F3: side quad
        [1, 3, 2],      # F4: bottom tri
        [4, 5, 6],      # F5: top tri
    ],
    num_edges=9, edge_type="Line2",
    edge_nodes=[
        (1, 2), (2, 3), (3, 1),    # bottom edges
        (1, 4), (2, 5), (3, 6),    # vertical edges
        (4, 5), (5, 6), (6, 4),    # top edges
    ],
    edge_groups=[("bottom", [0, 1, 2]), ("vertical", [3, 4, 5]), ("top", [6, 7, 8])],
    elevated_type="Prism18", num_elev_nodes=12,
    # CGNS ordering: bottom edges, TOP edges, VERTICAL edges, face centers
    elev_spans=[
        (1, 2), (2, 3), (3, 1),                    # bottom edges -> N7, N8, N9
        (4, 5), (5, 6), (6, 4),                    # top edges -> N10, N11, N12
        (1, 4), (2, 5), (3, 6),                    # vertical edges -> N13, N14, N15
        (1, 2, 5, 4), (2, 3, 6, 5), (3, 1, 4, 6), # face centers -> N16, N17, N18
    ],
    elev_span_types=["Line2"] * 9 + ["Quad4"] * 3,
    vtk_cell_type=13, vtk_node_order=[0, 1, 2, 3, 4, 5],
)

PRISM18 = ElementData(
    cgns_name="PENTA_18", dndsr_name="Prism18",
    dim=3, order=2, num_vertices=6, num_nodes=18,
    param_space="PrismSpace", param_space_vol="1.0",
    node_coords={
        # Corners
        1: (0, 0, -1), 2: (1, 0, -1), 3: (0, 1, -1),
        4: (0, 0, 1), 5: (1, 0, 1), 6: (0, 1, 1),
        # Bottom edge mids (E1-E3)
        7: (0.5, 0, -1), 8: (0.5, 0.5, -1), 9: (0, 0.5, -1),
        # Top edge mids (E7-E9) — CGNS puts these BEFORE vertical
        10: (0.5, 0, 1), 11: (0.5, 0.5, 1), 12: (0, 0.5, 1),
        # Vertical edge mids (E4-E6)
        13: (0, 0, 0), 14: (1, 0, 0), 15: (0, 1, 0),
        # Quad face centers
        16: (0.5, 0, 0), 17: (0.5, 0.5, 0), 18: (0, 0.5, 0),
    },
    num_faces=5,
    face_types=["Quad9", "Quad9", "Quad9", "Tri6", "Tri6"],
    face_nodes=[
        [1, 2, 5, 4, 7, 14, 10, 13, 16],    # F1: side quad9
        [2, 3, 6, 5, 8, 15, 11, 14, 17],    # F2: side quad9
        [3, 1, 4, 6, 9, 13, 12, 15, 18],    # F3: side quad9
        [1, 3, 2, 9, 8, 7],                  # F4: bottom tri6
        [4, 5, 6, 10, 11, 12],               # F5: top tri6
    ],
    num_edges=9, edge_type="Line3",
    edge_nodes=[
        (1, 2, 7), (2, 3, 8), (3, 1, 9),       # bottom
        (1, 4, 13), (2, 5, 14), (3, 6, 15),     # vertical
        (4, 5, 10), (5, 6, 11), (6, 4, 12),     # top
    ],
    edge_groups=[("bottom", [0, 1, 2]), ("vertical", [3, 4, 5]), ("top", [6, 7, 8])],
    num_bisect=8, num_bisect_variants=1,
    bisect_uniform_type="Prism6",
    # Bisect nodes use CGNS 1-based indexing
    bisect_elements=[
        (1, 7, 9, 13, 16, 18),
        (7, 2, 8, 16, 14, 17),
        (9, 7, 8, 18, 16, 17),
        (9, 8, 3, 18, 17, 15),
        (13, 16, 18, 4, 10, 12),
        (16, 14, 17, 10, 5, 11),
        (18, 16, 17, 12, 10, 11),
        (18, 17, 15, 12, 11, 6),
    ],
    vtk_cell_type=26,
    # VTK BiQuadraticQuadraticWedge: 15 nodes (no face centers)
    # VTK order: corners, bottom-edge-mids, top-edge-mids, vertical-edge-mids
    # (matches CGNS order for the first 15 nodes)
    vtk_node_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
)

HEX8 = ElementData(
    cgns_name="HEXA_8", dndsr_name="Hex8",
    dim=3, order=1, num_vertices=8, num_nodes=8,
    param_space="HexSpace", param_space_vol="8.0",
    node_coords={
        1: (-1, -1, -1), 2: (1, -1, -1), 3: (1, 1, -1), 4: (-1, 1, -1),
        5: (-1, -1, 1), 6: (1, -1, 1), 7: (1, 1, 1), 8: (-1, 1, 1),
    },
    num_faces=6,
    face_types=["Quad4"] * 6,
    face_nodes=[
        [1, 4, 3, 2],  # F1: bottom
        [1, 2, 6, 5],  # F2: front
        [2, 3, 7, 6],  # F3: right
        [3, 4, 8, 7],  # F4: back
        [1, 5, 8, 4],  # F5: left
        [5, 6, 7, 8],  # F6: top
    ],
    num_edges=12, edge_type="Line2",
    edge_nodes=[
        (1, 2), (2, 3), (3, 4), (4, 1),    # bottom
        (1, 5), (2, 6), (3, 7), (4, 8),    # vertical
        (5, 6), (6, 7), (7, 8), (8, 5),    # top
    ],
    edge_groups=[("bottom", [0, 1, 2, 3]), ("vertical", [4, 5, 6, 7]), ("top", [8, 9, 10, 11])],
    elevated_type="Hex27", num_elev_nodes=19,
    elev_spans=[
        (1, 2), (2, 3), (3, 4), (4, 1),              # bottom edges
        (1, 5), (2, 6), (3, 7), (4, 8),              # vertical edges
        (5, 6), (6, 7), (7, 8), (8, 5),              # top edges
        (1, 4, 3, 2), (1, 2, 6, 5),                  # face centers
        (2, 3, 7, 6), (3, 4, 8, 7),
        (1, 5, 8, 4), (5, 6, 7, 8),
        (1, 2, 3, 4, 5, 6, 7, 8),                    # body center
    ],
    elev_span_types=["Line2"] * 12 + ["Quad4"] * 6 + ["Hex8"],
    vtk_cell_type=12, vtk_node_order=[0, 1, 2, 3, 4, 5, 6, 7],
)

HEX27 = ElementData(
    cgns_name="HEXA_27", dndsr_name="Hex27",
    dim=3, order=2, num_vertices=8, num_nodes=27,
    param_space="HexSpace", param_space_vol="8.0",
    node_coords={
        1: (-1, -1, -1), 2: (1, -1, -1), 3: (1, 1, -1), 4: (-1, 1, -1),
        5: (-1, -1, 1), 6: (1, -1, 1), 7: (1, 1, 1), 8: (-1, 1, 1),
        # Edge mids
        9: (0, -1, -1), 10: (1, 0, -1), 11: (0, 1, -1), 12: (-1, 0, -1),
        13: (-1, -1, 0), 14: (1, -1, 0), 15: (1, 1, 0), 16: (-1, 1, 0),
        17: (0, -1, 1), 18: (1, 0, 1), 19: (0, 1, 1), 20: (-1, 0, 1),
        # Face centers
        21: (0, 0, -1), 22: (0, -1, 0), 23: (1, 0, 0),
        24: (0, 1, 0), 25: (-1, 0, 0), 26: (0, 0, 1),
        # Body center
        27: (0, 0, 0),
    },
    num_faces=6,
    face_types=["Quad9"] * 6,
    face_nodes=[
        [1, 4, 3, 2, 12, 11, 10, 9, 21],
        [1, 2, 6, 5, 9, 14, 17, 13, 22],
        [2, 3, 7, 6, 10, 15, 18, 14, 23],
        [3, 4, 8, 7, 11, 16, 19, 15, 24],
        [1, 5, 8, 4, 13, 20, 16, 12, 25],
        [5, 6, 7, 8, 17, 18, 19, 20, 26],
    ],
    num_edges=12, edge_type="Line3",
    edge_nodes=[
        (1, 2, 9), (2, 3, 10), (3, 4, 11), (4, 1, 12),
        (1, 5, 13), (2, 6, 14), (3, 7, 15), (4, 8, 16),
        (5, 6, 17), (6, 7, 18), (7, 8, 19), (8, 5, 20),
    ],
    edge_groups=[("bottom", [0, 1, 2, 3]), ("vertical", [4, 5, 6, 7]), ("top", [8, 9, 10, 11])],
    num_bisect=8, num_bisect_variants=1,
    bisect_uniform_type="Hex8",
    bisect_elements=[
        (1, 9, 21, 12, 13, 22, 27, 25),
        (9, 2, 10, 21, 22, 14, 23, 27),
        (12, 21, 11, 4, 25, 27, 24, 16),
        (21, 10, 3, 11, 27, 23, 15, 24),
        (13, 22, 27, 25, 5, 17, 26, 20),
        (22, 14, 23, 27, 17, 6, 18, 26),
        (25, 27, 24, 16, 20, 26, 19, 8),
        (27, 23, 15, 24, 26, 18, 7, 19),
    ],
    vtk_cell_type=25,
    # VTK TriQuadraticHexahedron: 20 nodes (no face/body centers)
    # VTK order: corners, bottom-edges, top-edges, vertical-edges
    vtk_node_order=[
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11,    # bottom edges
        16, 17, 18, 19,  # top edges (VTK puts these before vertical)
        12, 13, 14, 15,  # vertical edges
    ],
)


# ======================================================================
# Registry
# ======================================================================

ALL_ELEMENTS = [
    LINE2, LINE3,
    TRI3, TRI6,
    QUAD4, QUAD9,
    TET4, TET10,
    PYRAMID5, PYRAMID14,
    PRISM6, PRISM18,
    HEX8, HEX27,
]

BY_DNDSR_NAME = {e.dndsr_name: e for e in ALL_ELEMENTS}
BY_CGNS_NAME = {e.cgns_name: e for e in ALL_ELEMENTS}
