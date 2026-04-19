"""Line element shape function definitions."""

from .base import ElementDef, xi


class Line2Def(ElementDef):
    name = "Line2"
    dim = 1
    num_nodes = 2

    @classmethod
    def shape_functions(cls):
        return [
            (1 - xi) / 2,
            (1 + xi) / 2,
        ]


class Line3Def(ElementDef):
    name = "Line3"
    dim = 1
    num_nodes = 3

    @classmethod
    def shape_functions(cls):
        return [
            xi * (xi - 1) / 2,
            xi * (xi + 1) / 2,
            1 - xi**2,
        ]
