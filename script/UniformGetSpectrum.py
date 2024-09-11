import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser("UniformGetSpectrum")
parser.add_argument("inputname")
parser.add_argument("-o", "--out", required=False)
args = parser.parse_args()


fName = args.inputname
oName = args.out if args.out is not None else args.inputname + ".outspect"


file = h5py.File(name=fName, mode="r")
print(file["VTKHDF"].keys())

nPoint = file["VTKHDF"]["NumberOfPoints"]
nCell = file["VTKHDF"]["NumberOfCells"]
nConn = file["VTKHDF"]["NumberOfConnectivityIds"]
NGrid = int(np.round(np.power(nCell, 1 / 3)))
coords = file["VTKHDF"]["Points"]
velo = file["VTKHDF"]["PointData"]["Velo"]

print(f"nPoints {nPoint[0]}, nCells {nCell[0]}, nConn {nConn[0]}")
print(f"NGrid {NGrid}")

gSize = np.pi * 2 / NGrid

coordsIJK = np.round(np.array(coords) / gSize)
coordsIJKMax = np.max(coordsIJK)
coordsIJKMin = np.min(coordsIJK)
print(f"coordsIJKMin {coordsIJKMin}, coordsIJKMax {coordsIJKMax}")
print(coordsIJK[0:10, :])
coordsIJKInt = np.int64(coordsIJK)

VeloG = np.zeros(shape=(NGrid + 1, NGrid + 1, NGrid + 1, 3), dtype=np.double)
VeloG[coordsIJKInt[:, 0], coordsIJKInt[:, 1], coordsIJKInt[:, 2], 0:3] = velo

VeloGM = VeloG[0:NGrid, 0:NGrid, 0:NGrid, :]

VeloGM_K = np.fft.fftn(VeloGM, axes=[0, 1, 2]) / (NGrid**3)

print(VeloGM_K[0:4, 0:4, 0:4, :])

VeloGM_K = np.fft.fftshift(VeloGM_K, axes=[0, 1, 2])
indI = np.arange(0, NGrid, dtype=np.double)
kI = indI - (NGrid / 2)
kM = (
    np.reshape(kI, (1, 1, -1)) ** 2
    + np.reshape(kI, (1, -1, 1)) ** 2
    + np.reshape(kI, (-1, 1, 1)) ** 2
) ** 0.5

kMRound = np.int64(np.round(kM))
maxKMRound = np.max(kMRound)

EKGM_k = np.abs(VeloGM_K) ** 2


EKs = np.zeros((maxKMRound + 1, 3))
EKs_k = np.arange(0, maxKMRound + 1)
bins = np.arange(0, maxKMRound + 2)


(ek0, be) = np.histogram(kMRound, bins, density=False, weights=EKGM_k[:, :, :, 0])
(ek1, be) = np.histogram(kMRound, bins, density=False, weights=EKGM_k[:, :, :, 1])
(ek2, be) = np.histogram(kMRound, bins, density=False, weights=EKGM_k[:, :, :, 2])
print(ek0.shape)
EKs[:, 0] = ek0
EKs[:, 1] = ek1
EKs[:, 2] = ek2
# for i in range(maxKMRound):
#     EKs[i, 0] = np.sum(EKGM_k[kMRound == i, 0])
#     EKs[i, 1] = np.sum(EKGM_k[kMRound == i, 1])
#     EKs[i, 2] = np.sum(EKGM_k[kMRound == i, 2])
#     print(f"bin [{i}] done")
# for i in range(NGrid):
#     for j in range(NGrid):
#         for k in range(NGrid):
#             EKs[kMRound[i, j, k], :] += EKGM_k[i, j, k, :]
#     print(f"ind i [{i}] done")
print("Ek spectrum result: ")
EkResults = 0.5 * np.sum(EKs, axis=1)
print(EkResults)
np.savetxt(oName + ".txt", EkResults)


plt.plot(EKs_k[1:], 0.5 * np.sum(EKs[1:, :], axis=1))
plt.gca().set_yscale("log")
plt.gca().set_xscale("log")
plt.show()
