import h5py
import numpy as np
import copy, pprint


def read_vtkhdf_data(fName):
    ret = {}
    with h5py.File(name=fName, mode="r") as file:
        print(file["VTKHDF"].keys())
        print(file["VTKHDF"]["PointData"].keys())

        nPoint = file["VTKHDF"]["NumberOfPoints"]
        nCell = file["VTKHDF"]["NumberOfCells"]
        nConn = file["VTKHDF"]["NumberOfConnectivityIds"]

        ret["nPoint"] = int(nPoint[0])
        ret["nCell"] = int(nCell[0])
        ret["nConn"] = int(nConn[0])
        ret["Points"] = file["VTKHDF"]["Points"][:]
        ret["Types"] = file["VTKHDF"]["Types"][:]
        ret["Offsets"] = file["VTKHDF"]["Offsets"][:]
        ret["Connectivity"] = file["VTKHDF"]["Connectivity"][:]

        ret["PointData"] = {}
        for k in file["VTKHDF"]["PointData"].keys():
            ret["PointData"][k] = file["VTKHDF"]["PointData"][k][:]
        ret["CellData"] = {}
        for k in file["VTKHDF"]["CellData"].keys():
            ret["CellData"][k] = file["VTKHDF"]["CellData"][k][:]

        # ret["coords"] = int(coords)

    return ret

def check_vtkhdf_data_form(data):
    valid = True
    try:
        valid = valid and isinstance(data["nPoint"], int)
        valid = valid and isinstance(data["nCell"], int)
        valid = valid and isinstance(data["nConn"], int)
        valid = valid and "Points" in data
        valid = valid and "Types" in data
        valid = valid and "Offsets" in data
        valid = valid and "Connectivity" in data
        valid = valid and "PointData" in data
        valid = valid and "CellData" in data
    except BaseException as e:
        pprint.pprint(data)
        raise e
        
    return valid


def get_vtkhdf_cell_cent(data):
    data = copy.copy(data)
    data["CellCents"] = np.zeros((data["nCell"], 3)) * float("nan")
    cents = data["CellCents"]
    for iCell in range(data["nCell"]):
        off = data["Offsets"]
        c2n = data["Connectivity"][off[iCell] : off[iCell + 1]]
        cents[iCell, :] = np.mean(data["Points"][c2n, :], axis=0)
    return data

def get_triangle_vol(nodes: np.ndarray):
    return np.linalg.vector_norm(np.cross(nodes[1] - nodes[0], nodes[2]- nodes[0]))

def get_vtkhdf_elem_vol(type: int, nodes: np.ndarray):
    if type == 5: #Tri3
        return get_triangle_vol(nodes)
    if type == 9:  # Quad4
        return get_triangle_vol(nodes[[0, 1, 2]]) + get_triangle_vol(nodes[[0, 2, 3]])


def get_vtkhdf_cell_vol(data):
    data = copy.copy(data)
    data["CellVols"] = np.zeros((data["nCell"],)) * float("nan")
    vols = data["CellVols"]
    for iCell in range(data["nCell"]):
        off = data["Offsets"]
        c2n = data["Connectivity"][off[iCell] : off[iCell + 1]]
        vols[iCell] = get_vtkhdf_elem_vol(data["Types"][iCell], data["Points"][c2n])
    return data
