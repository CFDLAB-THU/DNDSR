"""Triangle element shape function definitions."""

from .base import ElementDef, xi, et
from sympy import Rational


class Tri3Def(ElementDef):
    name = "Tri3"
    dim = 2
    num_nodes = 3

    @classmethod
    def shape_functions(cls):
        L1 = 1 - xi - et  # barycentric coord for node 0
        return [L1, xi, et]


class Tri6Def(ElementDef):
    name = "Tri6"
    dim = 2
    num_nodes = 6

    @classmethod
    def shape_functions(cls):
        L1 = 1 - xi - et
        return [
            L1 * (2 * L1 - 1),        # node 0: vertex (0,0)
            xi * (2 * xi - 1),         # node 1: vertex (1,0)
            et * (2 * et - 1),         # node 2: vertex (0,1)
            4 * xi * L1,               # node 3: edge 0-1 midpoint
            4 * xi * et,               # node 4: edge 1-2 midpoint
            4 * et * L1,               # node 5: edge 2-0 midpoint
        ]
