function N = SF_Pyramid14(xi,et,zt)

NB = SF_Quad9(xi / (1-zt), et / (1-zt), 0);
NB_2 = SF_Quad4(xi / (1-zt), et  / (1-zt), 0);

Nz = SF_Line3(zt * 2 - 1, 0, 0);



N = [...
    NB(1) * Nz(1)
    NB(2) * Nz(1)
    NB(3) * Nz(1)
    NB(4) * Nz(1)
    Nz(2)
    NB(5) * Nz(1)
    NB(6) * Nz(1)
    NB(7) * Nz(1)
    NB(8) * Nz(1)
    NB_2(1) * Nz(3)
    NB_2(2) * Nz(3)
    NB_2(3) * Nz(3)
    NB_2(4) * Nz(3)
    NB(9) * Nz(1)
]';

end