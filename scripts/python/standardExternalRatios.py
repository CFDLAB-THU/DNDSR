LRatio = 1
RgD = 287.05
RInf = 1
pInf = float(input("p inf (NON dimensional) = "))
pInfD = float(input("p inf (dimensional) = "))
TInfD = float(input("T inf (dimensional) = "))
RInfD = pInfD / RgD / TInfD
RRatio = RInfD / RInf
pRatio = pInfD / pInf
vRatio = (pRatio / RRatio) ** 0.5
tRatio = LRatio / vRatio
ERatio = pRatio * LRatio ** 3
QRatio = ERatio / LRatio ** 2 / tRatio # heat flux

print(f"pRatio {pRatio}")
print(f"RRatio {RRatio}")
print(f"tRatio {tRatio}")
print(f"vRatio {vRatio}")
print(f"ERatio {ERatio}")
print(f"QRatio {QRatio}")
