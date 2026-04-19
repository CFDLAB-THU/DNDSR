function N = SF_Prism18(xi,et,zt)



NB = SF_Tri6(xi, et, 0);
Nz = SF_Line3(zt, 0, 0);

N = [...
   NB(1) * Nz(1)
   NB(2) * Nz(1)
   NB(3) * Nz(1)
   NB(1) * Nz(2)
   NB(2) * Nz(2)
   NB(3) * Nz(2)
   NB(4) * Nz(1) % 7
   NB(5) * Nz(1)
   NB(6) * Nz(1)
   NB(1) * Nz(3)
   NB(2) * Nz(3)
   NB(3) * Nz(3)
   NB(4) * Nz(2)
   NB(5) * Nz(2)
   NB(6) * Nz(2)
   NB(4) * Nz(3) %16
   NB(5) * Nz(3)
   NB(6) * Nz(3)
]';

end