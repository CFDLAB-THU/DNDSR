"""Prism (wedge) element shape function definitions."""

from .base import ElementDef, xi, et, zt


class Prism6Def(ElementDef):
    name = "Prism6"
    dim = 3
    num_nodes = 6

    @classmethod
    def shape_functions(cls):
        # Triangular base in (xi, et), linear in zt on [-1,1]
        L1 = 1 - xi - et
        Lz0 = (1 - zt) / 2
        Lz1 = (1 + zt) / 2
        return [
            L1 * Lz0,   # node 0
            xi * Lz0,   # node 1
            et * Lz0,   # node 2
            L1 * Lz1,   # node 3
            xi * Lz1,   # node 4
            et * Lz1,   # node 5
        ]


class Prism18Def(ElementDef):
    name = "Prism18"
    dim = 3
    num_nodes = 18

    @classmethod
    def shape_functions(cls):
        # Quadratic triangle in (xi, et) x quadratic line in zt on [-1,1]
        L1 = 1 - xi - et

        # Quadratic triangle basis (6 nodes)
        T = [
            L1 * (2 * L1 - 1),
            xi * (2 * xi - 1),
            et * (2 * et - 1),
            4 * xi * L1,
            4 * xi * et,
            4 * et * L1,
        ]

        # 1D quadratic basis on [-1,1]
        Lz = [
            zt * (zt - 1) / 2,
            zt * (zt + 1) / 2,
            1 - zt**2,
        ]

        # Prism18 node ordering (CGNS):
        # 0-2: bottom tri vertices (zt=-1) -> T[0..2] * Lz[0]
        # 3-5: top tri vertices (zt=+1) -> T[0..2] * Lz[1]
        # 6-8: bottom tri edges -> T[3..5] * Lz[0]
        # 9-11: mid-height vertices -> T[0..2] * Lz[2]
        # 12-14: top tri edges -> T[3..5] * Lz[1]
        # 15-17: mid-height edges -> T[3..5] * Lz[2]
        return [
            T[0] * Lz[0],   # 0
            T[1] * Lz[0],   # 1
            T[2] * Lz[0],   # 2
            T[0] * Lz[1],   # 3
            T[1] * Lz[1],   # 4
            T[2] * Lz[1],   # 5
            T[3] * Lz[0],   # 6
            T[4] * Lz[0],   # 7
            T[5] * Lz[0],   # 8
            T[0] * Lz[2],   # 9
            T[1] * Lz[2],   # 10
            T[2] * Lz[2],   # 11
            T[3] * Lz[1],   # 12
            T[4] * Lz[1],   # 13
            T[5] * Lz[1],   # 14
            T[3] * Lz[2],   # 15
            T[4] * Lz[2],   # 16
            T[5] * Lz[2],   # 17
        ]
