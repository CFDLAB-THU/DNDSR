function N = SF_Prism6(xi,et,zt)



NB = SF_Tri3(xi, et, 0);
Nz = SF_Line2(zt, 0, 0);

N = [...
   NB(1) * Nz(1)
   NB(2) * Nz(1)
   NB(3) * Nz(1)
   NB(1) * Nz(2)
   NB(2) * Nz(2)
   NB(3) * Nz(2)
]';

end