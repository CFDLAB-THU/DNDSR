"""Quadrilateral element shape function definitions."""

from .base import ElementDef, xi, et


class Quad4Def(ElementDef):
    name = "Quad4"
    dim = 2
    num_nodes = 4

    @classmethod
    def shape_functions(cls):
        # Nodes at (-1,-1), (1,-1), (1,1), (-1,1)
        xi_i = [-1, 1, 1, -1]
        et_i = [-1, -1, 1, 1]
        return [(1 + xi_i[j] * xi) * (1 + et_i[j] * et) / 4 for j in range(4)]


class Quad9Def(ElementDef):
    name = "Quad9"
    dim = 2
    num_nodes = 9

    @classmethod
    def shape_functions(cls):
        # 1D quadratic Lagrange basis on [-1,1]:
        # L0(t) = t*(t-1)/2, L1(t) = t*(t+1)/2, L2(t) = 1-t^2
        # Node ordering (CGNS): corners 0-3, edge midpoints 4-7, center 8
        # xi_i, et_i for each node:
        xi_nodes = [-1, 1, 1, -1, 0, 1, 0, -1, 0]
        et_nodes = [-1, -1, 1, 1, -1, 0, 1, 0, 0]

        def L(t, ti):
            """1D quadratic Lagrange basis value at node position ti."""
            if ti == -1:
                return t * (t - 1) / 2
            elif ti == 1:
                return t * (t + 1) / 2
            else:  # ti == 0
                return 1 - t**2

        return [L(xi, xi_nodes[j]) * L(et, et_nodes[j]) for j in range(9)]
