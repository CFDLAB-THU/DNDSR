import math

Ma = float(input("Mach Number = "))
Re = float(input("Re Number = "))
alpha = float(input("AoA in degrees = "))
LRef = float(input("length ref = "))
wingDir = str(input("Wing Direction, z or y: "))

gamma = 1.4
vInf = 1
rInf = 1
TInf = 273.15

a = vInf / Ma
p = a**2 / gamma / rInf
E = p / (gamma - 1) + 0.5 * rInf * vInf**2
rg = p / rInf / TInf
muGas = rInf * vInf * LRef / Re

vX = vInf * math.cos(alpha / 180 * math.pi)
vY = vInf * math.sin(alpha / 180 * math.pi)
UInf = [rInf, vX * rInf, vY if wingDir == "z" else 0, 0 if wingDir == "z" else vY, E]

print(f"UInf = {UInf},\n Rgas = {rg}, muGas = {muGas}")
