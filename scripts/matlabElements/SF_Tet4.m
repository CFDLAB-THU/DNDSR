function N = SF_Tet4(xi,et,zt)
a = 1 - xi - et - zt;
b = xi;
c = et;
d = zt;
N = [...
   a
   b
   c
   d
]';

end