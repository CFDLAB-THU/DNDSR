"""Tetrahedron element shape function definitions."""

from .base import ElementDef, xi, et, zt


class Tet4Def(ElementDef):
    name = "Tet4"
    dim = 3
    num_nodes = 4

    @classmethod
    def shape_functions(cls):
        L1 = 1 - xi - et - zt
        return [L1, xi, et, zt]


class Tet10Def(ElementDef):
    name = "Tet10"
    dim = 3
    num_nodes = 10

    @classmethod
    def shape_functions(cls):
        L1 = 1 - xi - et - zt
        return [
            L1 * (2 * L1 - 1),        # node 0
            xi * (2 * xi - 1),         # node 1
            et * (2 * et - 1),         # node 2
            zt * (2 * zt - 1),         # node 3
            4 * xi * L1,               # node 4: edge 0-1
            4 * xi * et,               # node 5: edge 1-2
            4 * et * L1,               # node 6: edge 2-0
            4 * zt * L1,               # node 7: edge 0-3
            4 * xi * zt,               # node 8: edge 1-3
            4 * et * zt,               # node 9: edge 2-3
        ]
