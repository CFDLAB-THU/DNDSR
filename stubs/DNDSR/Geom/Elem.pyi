from __future__ import annotations
import typing
__all__: list[str] = ['ElemType']
class ElemType:
    """
    Members:
    
      UnknownElem
    
      Line2
    
      Line3
    
      Tri3
    
      Tri6
    
      Quad4
    
      Quad9
    
      Tet4
    
      Tet10
    
      Hex8
    
      Hex27
    
      Prism6
    
      Prism18
    
      Pyramid5
    
      Pyramid14
    """
    Hex27: typing.ClassVar[ElemType]  # value = <ElemType.Hex27: 12>
    Hex8: typing.ClassVar[ElemType]  # value = <ElemType.Hex8: 5>
    Line2: typing.ClassVar[ElemType]  # value = <ElemType.Line2: 1>
    Line3: typing.ClassVar[ElemType]  # value = <ElemType.Line3: 8>
    Prism18: typing.ClassVar[ElemType]  # value = <ElemType.Prism18: 13>
    Prism6: typing.ClassVar[ElemType]  # value = <ElemType.Prism6: 6>
    Pyramid14: typing.ClassVar[ElemType]  # value = <ElemType.Pyramid14: 14>
    Pyramid5: typing.ClassVar[ElemType]  # value = <ElemType.Pyramid5: 7>
    Quad4: typing.ClassVar[ElemType]  # value = <ElemType.Quad4: 3>
    Quad9: typing.ClassVar[ElemType]  # value = <ElemType.Quad9: 10>
    Tet10: typing.ClassVar[ElemType]  # value = <ElemType.Tet10: 11>
    Tet4: typing.ClassVar[ElemType]  # value = <ElemType.Tet4: 4>
    Tri3: typing.ClassVar[ElemType]  # value = <ElemType.Tri3: 2>
    Tri6: typing.ClassVar[ElemType]  # value = <ElemType.Tri6: 9>
    UnknownElem: typing.ClassVar[ElemType]  # value = <ElemType.UnknownElem: 0>
    __members__: typing.ClassVar[dict[str, ElemType]]  # value = {'UnknownElem': <ElemType.UnknownElem: 0>, 'Line2': <ElemType.Line2: 1>, 'Line3': <ElemType.Line3: 8>, 'Tri3': <ElemType.Tri3: 2>, 'Tri6': <ElemType.Tri6: 9>, 'Quad4': <ElemType.Quad4: 3>, 'Quad9': <ElemType.Quad9: 10>, 'Tet4': <ElemType.Tet4: 4>, 'Tet10': <ElemType.Tet10: 11>, 'Hex8': <ElemType.Hex8: 5>, 'Hex27': <ElemType.Hex27: 12>, 'Prism6': <ElemType.Prism6: 6>, 'Prism18': <ElemType.Prism18: 13>, 'Pyramid5': <ElemType.Pyramid5: 7>, 'Pyramid14': <ElemType.Pyramid14: 14>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
