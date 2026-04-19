import numpy as np


def getPolyOrder():
    dol2D = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [2, 0, 0],
        [1, 1, 0],
        [0, 2, 0],
        [3, 0, 0],
        [2, 1, 0],
        [1, 2, 0],
        [0, 3, 0],
        [4, 0, 0],
        [3, 1, 0],
        [2, 2, 0],
        [1, 3, 0],
        [0, 4, 0],
    ]
    dol3D = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 3],
        [2, 1, 0],
        [1, 2, 0],
        [0, 2, 1],
        [0, 1, 2],
        [1, 0, 2],
        [2, 0, 1],
        [1, 1, 1],
        [4, 0, 0],
        [0, 4, 0],
        [0, 0, 4],
        [3, 1, 0],
        [0, 3, 1],
        [1, 0, 3],
        [1, 3, 0],
        [0, 1, 3],
        [3, 0, 1],
        [2, 2, 0],
        [0, 2, 2],
        [2, 0, 2],
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2],
    ]

    dol2Dn = np.array(dol2D, dtype=np.int32)
    dol3Dn = np.array(dol3D, dtype=np.int32)
    return (dol2Dn, dol3Dn)


def polyOrder2Name(polyOrders):
    names2D = []
    names3D = []
    for row in polyOrders[0]:
        strc = ""
        for i in range(3):
            strc = strc + str(i) * row[i]
        names2D.append(strc)

    for row in polyOrders[1]:
        strc = ""
        for i in range(3):
            strc = strc + str(i) * row[i]
        names3D.append(strc)
    return (names2D, names3D)


def searchRow(dols, diffRow):
    nSearch = 0
    iSearch = -1
    for iS in range(dols.shape[0]):
        if (dols[iS] == diffRow).all():
            nSearch += 1
            iSearch = iS
    
    return (nSearch == 1, iSearch)
