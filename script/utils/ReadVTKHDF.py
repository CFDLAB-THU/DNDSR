import h5py
import numpy as np


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
