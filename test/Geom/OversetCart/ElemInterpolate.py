from DNDSR import DNDS, Geom
import numpy as np
import numba


def elem_get_interpolation_base(type: Geom.Elem.ElemType, coords: np.ndarray, xq):
    """
    Note:
    xq must be internal
    """
    if type in [Geom.Elem.ElemType.Tri3, Geom.Elem.ElemType.Quad4]:
        nv = 3 if type == Geom.Elem.ElemType.Tri3 else 4

        xRel = coords.T - xq
        areas = np.zeros(nv)
        for i in range(nv):
            areas[i] = np.linalg.norm(np.cross(xRel[i], xRel[(i + 1) % nv]))
        if type == Geom.Elem.ElemType.Tri3:
            # area = [A2, A0, A1]
            return np.roll(areas, -1) / areas.sum()
        if type == Geom.Elem.ElemType.Quad4:
            vxi = areas[3] / (areas[1] + areas[3])
            vet = areas[0] / (areas[0] + areas[2])
            return np.array(
                [(1 - vxi) * (1 - vet), vxi * (1 - vet), vxi * vet, (1 - vxi) * vet]
            )
    else:
        raise ValueError(f"type not supported: {type}")


if __name__ == "__main__":
    coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64).T
    print(
        elem_get_interpolation_base(
            Geom.Elem.ElemType.Tri3, coords, np.array([0.1, 0.1, 0])
        )
    )
    print(
        elem_get_interpolation_base(
            Geom.Elem.ElemType.Quad4, coords, np.array([0.1, 0.1, 0])
        )
    )
