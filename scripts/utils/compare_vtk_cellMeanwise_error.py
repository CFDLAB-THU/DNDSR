import os, sys

# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
if __name__ == "__main__":
    from ReadVTKHDF import read_vtkhdf_data, get_vtkhdf_cell_cent, get_vtkhdf_cell_vol, check_vtkhdf_data_form
else:
    from .ReadVTKHDF import read_vtkhdf_data, get_vtkhdf_cell_cent, get_vtkhdf_cell_vol, check_vtkhdf_data_form
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pprint


def detect_mesh_inconsistency(data1, data2, coord_validate_threshold=1e-15):
    if (
        (data1["nPoint"] != data2["nPoint"])
        or (data1["nCell"] != data2["nCell"])
        or (data1["nConn"] != data2["nConn"])
    ):
        raise ValueError(
            f"vtkhdf mesh size not consistent: \n"
            + pprint.pformat(data1)
            + "\n"
            + pprint.pformat(data2)
            + "\n"
        )
    type_difference = np.abs(
        data1["Types"].astype(np.int64) - data2["Types"].astype(np.int64)
    ).sum()

    if type_difference != 0:
        raise ValueError(
            f"vtkhdf mesh cell types not consistent {type_difference}: \n"
            + pprint.pformat(data1)
            + "\n"
            + pprint.pformat(data2)
            + "\n"
        )

    conn_difference = np.abs(
        data1["Connectivity"].astype(np.int64) - data2["Connectivity"].astype(np.int64)
    ).sum()

    if conn_difference != 0:
        raise ValueError(
            f"vtkhdf mesh cell types not consistent {conn_difference}: \n"
            + pprint.pformat(data1)
            + "\n"
            + pprint.pformat(data2)
            + "\n"
        )

    point_difference = np.abs(
        data1["Points"].astype(np.float64) - data2["Points"].astype(np.float64)
    ).sum()

    if point_difference > coord_validate_threshold:
        raise ValueError(
            f"vtkhdf mesh cell types not consistent {point_difference}: \n"
            + pprint.pformat(data1)
            + "\n"
            + pprint.pformat(data2)
            + "\n"
        )


def get_ordering_by_point_geom(data1, data2, coord_validate_threshold=1e-15):
    raise NotImplementedError()


def compare_vtkhdf_data_cellwise(
    f1: str | dict,
    f2: str | dict,
    field="R",
    on_cell=True,
    cell_vol=False,
    coord_validate_threshold=1e-15,
    norm_ord="L1",
    mask=lambda point: np.ones((point.shape[0],), dtype=np.float64),
):
    
    if isinstance(f1, str):
        if not f1.endswith("vtkhdf"):
            raise ValueError("has to be vtkhdf")
        data1 = read_vtkhdf_data(f1)
    elif check_vtkhdf_data_form(f1):
        data1 = f1
    else:
        raise ValueError("f1 not valid type")
    
    if isinstance(f2, str):
        if not f2.endswith("vtkhdf"):
            raise ValueError("has to be vtkhdf")
        data2 = read_vtkhdf_data(f2)
    
        # data1 = get_vtkhdf_cell_cent(data1)
        data2 = get_vtkhdf_cell_cent(data2)
        if cell_vol:
            data2 = get_vtkhdf_cell_vol(data2)
            # data1 = get_vtkhdf_cell_vol(data1)
    elif check_vtkhdf_data_form(f2):
        data2 = f2
    else:
        raise ValueError("f2 not valid type")

    field_type = "CellData" if on_cell else "PointData"

    n_points_1 = data1[field_type][field].shape[0]
    n_points_2 = data2[field_type][field].shape[0]
    assert n_points_1 == n_points_2
    data1Data = data1[field_type][field]
    data2Data = data2[field_type][field]

    try:
        detect_mesh_inconsistency(data1, data2, coord_validate_threshold)
    except ValueError as e:
        print("Trying reordering, Error detected is " + str(e))
        data2Data = data2Data[
            get_ordering_by_point_geom(data1, data2, coord_validate_threshold), ...
        ]

    reshape_size = (-1,) + (1,) * (data1Data.ndim - 1)
    errArr = (data1Data - data2Data) * np.reshape(
        mask(data2["CellCents"] if on_cell else data2["Points"]),
        reshape_size,
    )
    norm_tot = n_points_1

    if on_cell and cell_vol:
        errArr *= np.reshape(data2["CellVols"], reshape_size)
        norm_tot = 1
        print(f"Volume Max {data2["CellVols"].max()} Min {data2["CellVols"].min()}")

    # errArr = data1Data - data2Data
    

    error = None
    if norm_ord.lower() == "l1":
        error = np.linalg.vector_norm(errArr, axis=0, ord=1) / norm_tot
    elif norm_ord.lower() == "l2":
        error = (np.linalg.vector_norm(errArr, axis=0, ord=2) ** 2 / norm_tot) ** 0.5
    elif norm_ord.lower() == "linf":
        error = np.linalg.vector_norm(errArr, axis=0, ord=np.inf)
    else:
        raise ValueError(f"not supporing {norm_ord}")
    
    return error, data1, data2


if __name__ == "__main__":
    parser = argparse.ArgumentParser("compare_vtk_cellMeanwise_error")
    parser.add_argument("f1")
    parser.add_argument("f2")
    args = parser.parse_args()

    pprint.pprint(compare_vtkhdf_data_cellwise(args.f1, args.f2))
