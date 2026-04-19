"""Hexahedron element shape function definitions."""

from .base import ElementDef, xi, et, zt


class Hex8Def(ElementDef):
    name = "Hex8"
    dim = 3
    num_nodes = 8

    @classmethod
    def shape_functions(cls):
        # Nodes at corners of [-1,1]^3 in CGNS order
        xi_i = [-1, 1, 1, -1, -1, 1, 1, -1]
        et_i = [-1, -1, 1, 1, -1, -1, 1, 1]
        zt_i = [-1, -1, -1, -1, 1, 1, 1, 1]
        return [
            (1 + xi_i[j] * xi) * (1 + et_i[j] * et) * (1 + zt_i[j] * zt) / 8
            for j in range(8)
        ]


class Hex27Def(ElementDef):
    name = "Hex27"
    dim = 3
    num_nodes = 27

    @classmethod
    def shape_functions(cls):
        # CGNS Hex27 node coordinates (0-based)
        xi_n = [-1, 1, 1, -1, -1, 1, 1, -1,
                 0, 1, 0, -1, -1, 1, 1, -1,
                 0, 1, 0, -1, 0, 0, 1, 0, -1, 0, 0]
        et_n = [-1, -1, 1, 1, -1, -1, 1, 1,
                -1, 0, 1, 0, -1, -1, 1, 1,
                -1, 0, 1, 0, 0, -1, 0, 1, 0, 0, 0]
        zt_n = [-1, -1, -1, -1, 1, 1, 1, 1,
                -1, -1, -1, -1, 0, 0, 0, 0,
                 1, 1, 1, 1, -1, 0, 0, 0, 0, 1, 0]

        def L(t, ti):
            if ti == -1:
                return t * (t - 1) / 2
            elif ti == 1:
                return t * (t + 1) / 2
            else:
                return 1 - t**2

        return [
            L(xi, xi_n[j]) * L(et, et_n[j]) * L(zt, zt_n[j])
            for j in range(27)
        ]
