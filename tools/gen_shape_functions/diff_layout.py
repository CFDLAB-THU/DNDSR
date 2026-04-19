"""
Derivative row layout definitions.

Defines how partial derivatives map to row indices in the v(row, col) output
matrix for each diffOrder and spatial dimension.  These must match the
BaseFunction.hpp definitions (diffOperatorOrderList, diffOperatorOrderList2D).
"""

# 1D: each diffOrder has exactly 1 row
# row 0 = the single derivative of that order

# 2D layout (from diffOperatorOrderList2D in BaseFunction.hpp):
#   diffOrder 1: row 0 = d/dxi, row 1 = d/det
#   diffOrder 2: row 0 = d2/dxi2, row 1 = d2/dxidet, row 2 = d2/det2
#   diffOrder 3: row 0 = d3/dxi3, row 1 = d3/dxi2det, row 2 = d3/dxidet2, row 3 = d3/det3
DIFF_ORDER_2D = {
    1: [(1, 0), (0, 1)],
    2: [(2, 0), (1, 1), (0, 2)],
    3: [(3, 0), (2, 1), (1, 2), (0, 3)],
}

# 3D layout (from diffOperatorOrderList in BaseFunction.hpp):
#   diffOrder 1: row 0 = d/dxi, row 1 = d/det, row 2 = d/dzt
#   diffOrder 2: row 0 = d2/dxi2, row 1 = d2/det2, row 2 = d2/dzt2,
#                row 3 = d2/dxidet, row 4 = d2/detdzt, row 5 = d2/dxidzt
#   diffOrder 3: row 0 = (3,0,0), row 1 = (0,3,0), row 2 = (0,0,3),
#                row 3 = (2,1,0), row 4 = (1,2,0), row 5 = (0,2,1),
#                row 6 = (0,1,2), row 7 = (1,0,2), row 8 = (2,0,1),
#                row 9 = (1,1,1)
DIFF_ORDER_3D = {
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (0, 1, 1), (1, 0, 1)],
    3: [
        (3, 0, 0), (0, 3, 0), (0, 0, 3),
        (2, 1, 0), (1, 2, 0), (0, 2, 1),
        (0, 1, 2), (1, 0, 2), (2, 0, 1),
        (1, 1, 1),
    ],
}


def get_diff_layout(dim, diff_order):
    """Return the list of derivative tuples for a given dimension and diff order.

    Each tuple has length `dim` and represents the partial derivative powers.
    E.g. (2,1,0) means d^3 / (dxi^2 det).

    For diffOrder==0, returns [tuple of zeros].
    """
    if diff_order == 0:
        return [tuple(0 for _ in range(max(dim, 1)))]

    if dim == 1:
        return [(diff_order,)]
    elif dim == 2:
        return DIFF_ORDER_2D[diff_order]
    elif dim == 3:
        return DIFF_ORDER_3D[diff_order]
    else:
        raise ValueError(f"Unsupported dim={dim}")


def num_rows(dim, diff_order):
    """Number of derivative rows for a given dimension and diff order."""
    return len(get_diff_layout(dim, diff_order))
