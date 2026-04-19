import numpy as np

# import numba
import copy
import timeit


def segments_intersect_polygon_edge(polygon, x0, x1, y0, y1, closed=True):
    assert polygon.ndim == 2

    # Polygon edges
    if closed:
        xp0, yp0 = polygon
        xp1, yp1 = np.roll(xp0, 1), np.roll(yp0, 1)
    else:
        xp0, yp0 = polygon[:, :-1]
        xp1, yp1 = polygon[:, 1:]

    # Broadcast query segments
    x0 = np.asarray(x0)[..., None]
    y0 = np.asarray(y0)[..., None]
    x1 = np.asarray(x1)[..., None]
    y1 = np.asarray(y1)[..., None]

    # Vectorized segment to segment intersection
    # Segments: (x0,y0)-(x1,y1) and (xp0,yp0)-(xp1,yp1)

    # Segment directions
    dx1 = x1 - x0
    dy1 = y1 - y0
    dx2 = xp1 - xp0
    dy2 = yp1 - yp0

    # Denominator for parametric equations
    denom = dx1 * dy2 - dy1 * dx2
    denominatorScale = dx2**2 + dy2**2 + dx1**2 + dy1**2

    # Parallel segments (denom == 0) won't intersect
    parallel = np.isclose(denom, 0, atol=denominatorScale * 1e-8)
    denom[parallel] = 1e300

    # Compute numerators
    dx0 = xp0 - x0
    dy0 = yp0 - y0

    t = (dx0 * dy2 - dy0 * dx2) / denom
    u = (dx0 * dy1 - dy0 * dx1) / denom

    # Conditions for proper intersection: t, u in [0, 1]
    intersects = (~parallel) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)

    # Now, for each query segment, check if any intersection
    any_intersect = np.any(intersects, axis=-1)
    # print(polygon)
    # print(x1)

    return any_intersect


def points_in_polygon_winding(polygon, x, y, closed=True):
    assert polygon.ndim == 2
    if closed:
        xp0, yp0 = polygon
        xp1, yp1 = np.roll(xp0, 1), np.roll(yp0, 1)
    else:
        xp0, yp0 = polygon[:, :-1]
        xp1, yp1 = polygon[:, 1:]

    x = np.asarray(x)[..., None]  # Add axis for broadcasting
    y = np.asarray(y)[..., None]

    # Vectors from point to edge start and end
    x0, y0 = xp0 - x, yp0 - y
    x1, y1 = xp1 - x, yp1 - y

    # Compute angles between vectors (x0, y0) and (x1, y1)
    cross = x0 * y1 - x1 * y0  # determinant
    dot = x0 * x1 + y0 * y1  # dot product
    angles = np.arctan2(cross, dot)
    # when some angle is close to ±π, or to say cross is very close to 0, it's on the edge
    on_some_edge = np.isclose(np.abs(angles), np.pi, rtol=1e-9).any(axis=-1)

    # Sum of angles gives winding number
    winding_number = np.sum(angles, axis=-1)

    # Inside if winding number is approximately ±2π
    return (np.abs(winding_number) > 1e-8) | on_some_edge


def single_elem_intersection_mask_2D(
    origin: np.ndarray, h: float, ijks: np.ndarray, elem: np.ndarray, closed=True
):
    """
    Parameters:
    origin: (2,) array
    h: float
    ijks: (2, ...) integer array
    elem: is a 2D polygon, 2D coords of shape (2+, n_point)
    """
    assert elem.ndim == 2
    n_point = elem.shape[1]
    originCS = np.ones(ijks.ndim, dtype=np.int64)
    originCS[0] = 2
    ll = origin.reshape(originCS) + ijks.astype(np.float64) * h
    ru = origin.reshape(originCS) + (ijks.astype(np.float64) + 1) * h

    cshapeB = np.array(ru.shape, dtype=np.int64)
    cshapeB[0] = 1
    eshapeB = np.ones(ijks.ndim, dtype=np.int64)
    eshapeB[0] = n_point

    # print(ru[0])

    elemXViewB = elem[0].reshape(eshapeB)
    elemYViewB = elem[1].reshape(eshapeB)

    xMaxViewB = ru[0].reshape(cshapeB)
    xMinViewB = ll[0].reshape(cshapeB)
    yMaxViewB = ru[1].reshape(cshapeB)
    yMinViewB = ll[1].reshape(cshapeB)

    any_e_node_in_box = (
        (elemXViewB <= xMaxViewB)
        & (elemXViewB >= xMinViewB)
        & (elemYViewB <= yMaxViewB)
        & (elemYViewB >= yMinViewB)
    ).any(axis=0)

    any_b_node_in_elem = (
        points_in_polygon_winding(elem, ll[0], ll[1], closed)
        | points_in_polygon_winding(elem, ll[0], ru[1], closed)
        | points_in_polygon_winding(elem, ru[0], ll[1], closed)
        | points_in_polygon_winding(elem, ru[0], ru[1], closed)
    )

    any_b_edge_v_elem_edge = (  # mind the order: x0 x1 y0 y1
        segments_intersect_polygon_edge(elem, ll[0], ru[0], ll[1], ll[1], closed)
        | segments_intersect_polygon_edge(elem, ll[0], ru[0], ru[1], ru[1], closed)
        | segments_intersect_polygon_edge(elem, ll[0], ll[0], ll[1], ru[1], closed)
        | segments_intersect_polygon_edge(elem, ru[0], ru[0], ll[1], ru[1], closed)
    )

    # print(any_b_node_in_elem)
    return any_b_node_in_elem | any_e_node_in_box | any_b_edge_v_elem_edge


def single_elem_grid_points_mask_2D(
    origin: np.ndarray, h: float, ijks: np.ndarray, elem: np.ndarray, closed=True
):
    """
    Parameters:
    origin: (2,) array
    h: float
    ijks: (2, ...) integer array
    elem: is a 2D polygon, 2D coords of shape (2+, n_point)
    """
    assert elem.ndim == 2
    n_point = elem.shape[1]
    originCS = np.ones(ijks.ndim, dtype=np.int64)
    originCS[0] = 2
    ll = origin.reshape(originCS) + ijks.astype(np.float64) * h

    cshapeB = np.array(ll.shape, dtype=np.int64)
    cshapeB[0] = 1
    eshapeB = np.ones(ijks.ndim, dtype=np.int64)
    eshapeB[0] = n_point

    return points_in_polygon_winding(elem, ll[0], ll[1], closed)


def single_elem_box_range_2D(
    origin: np.ndarray, h: float, elem: np.ndarray, grid_eps=1e-10
):
    xyzmin = elem.min(axis=1)
    xyzmax = elem.max(axis=1)

    lowerijk = np.floor((xyzmin - origin) / h - grid_eps)
    upperijk = np.ceil((xyzmax - origin) / h + grid_eps) #add eps here to avoid problems when elem is precisely edge

    assert np.all(
        (origin + upperijk * h) + grid_eps >= xyzmax
    ), f"{origin}, {upperijk}, {origin + upperijk * h},\n {elem}\n, {xyzmax}"
    assert np.all((origin + lowerijk * h) - grid_eps <= xyzmin)

    return (np.asarray(lowerijk, dtype=np.int64), np.asarray(upperijk, dtype=np.int64))


def single_elem_get_box_intersection_2D(
    origin: np.ndarray, h: float, elem: np.ndarray, closed=True
):
    """get element intersecting boxes

    Args:
        origin (np.ndarray): _description_
        h (float): _description_
        elem (np.ndarray): _description_
        closed (bool, optional): _description_. Defaults to True.

    Returns:
        np.ndarray: ijks of the touched cells
    """
    lowerijk, upperijk = single_elem_box_range_2D(origin, h, elem)
    iss = np.arange(lowerijk[0], upperijk[0], dtype=np.int64)
    jss = np.arange(lowerijk[1], upperijk[1], dtype=np.int64)
    ism, jsm = np.meshgrid(iss, jss)
    ijks = np.array([ism, jsm])
    mask = single_elem_intersection_mask_2D(origin, h, ijks, elem, closed=closed)
    return ijks[:, mask]


def single_elem_get_grid_point_2D(
    origin: np.ndarray, h: float, elem: np.ndarray, closed=True
):
    """
    get grid point ijks inside the element
    Args:
        origin (np.ndarray): _description_
        h (float): _description_
        elem (np.ndarray): _description_
        closed (bool, optional): _description_. Defaults to True.

    Returns:
        np.ndarray: _description_
    """
    lowerijk, upperijk = single_elem_box_range_2D(origin, h, elem)
    iss = np.arange(lowerijk[0], upperijk[0] + 1, dtype=np.int64)
    jss = np.arange(lowerijk[1], upperijk[1] + 1, dtype=np.int64)
    ism, jsm = np.meshgrid(iss, jss)
    ijks = np.array([ism, jsm])

    mask = single_elem_grid_points_mask_2D(origin, h, ijks, elem, closed=closed)
    return ijks[:, mask]


def points_get_grid_cells_2D(origin: np.ndarray, h: float, coords: np.ndarray):
    originCS = np.ones(coords.ndim, dtype=np.int64)
    originCS[0] = coords.shape[0]
    lowerijk = np.floor((coords - origin.reshape(originCS)) / h)
    return np.array(lowerijk, dtype=np.int64)


if __name__ == "__main__":

    elem_test = np.array([[0.5, 2.5, 0.5], [0.5, 0.5, 2.5]], dtype=np.float64)
    elem_test_1 = np.array([[0.5, 2.5, 0.5], [0.5, 0.5, 0.6]], dtype=np.float64)
    origin_test = np.array([0.0, 0.0], dtype=np.float64)
    print(
        single_elem_intersection_mask_2D(
            origin_test,
            1.0,
            np.array(
                [[0, 1, 2, 3] * 4, [i // 4 for i in range(4 * 4)]], dtype=np.int64
            ).reshape((2, 4, 4)),
            elem_test,
        )
    )

    assert (
        single_elem_get_box_intersection_2D(
            origin_test,
            1.0,
            elem_test_1,
        ).shape[1]
        == 3
    )

    time_result = timeit.timeit(
        lambda: single_elem_get_box_intersection_2D(origin_test, 1.0, elem_test),
        number=100,
    )
    print(f"time: {time_result}")

    print(single_elem_get_box_intersection_2D(origin_test, 1.0, elem_test))
    print(single_elem_get_grid_point_2D(origin_test, 1.0, elem_test))

    # print(
    #     single_elem_intersection_2D(
    #         np.array((0.0, 0.0)),
    #         1.0,
    #         np.array([[0], [4]]).reshape((2, 1, 1)),
    #         np.array([[0.5, 2.5, 0.5], [0.5, 0.5, 2.5]]),
    #     )
    # )
