"""Element shape function definitions for code generation."""

from .line import Line2Def, Line3Def
from .triangle import Tri3Def, Tri6Def
from .quad import Quad4Def, Quad9Def
from .tet import Tet4Def, Tet10Def
from .hex import Hex8Def, Hex27Def
from .prism import Prism6Def, Prism18Def
from .pyramid import Pyramid5Def, Pyramid14Def

ALL_ELEMENTS = [
    Line2Def, Line3Def,
    Tri3Def, Tri6Def,
    Quad4Def, Quad9Def,
    Tet4Def, Tet10Def,
    Hex8Def, Hex27Def,
    Prism6Def, Prism18Def,
    Pyramid5Def, Pyramid14Def,
]
