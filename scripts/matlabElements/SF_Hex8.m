function N = SF_Hex8(xi,et,zt)

Nxi = SF_Line2(xi, 0, 0);
Net = SF_Line2(et, 0, 0);
Nzt = SF_Line2(zt, 0, 0);

N = [...
   Nxi(1) * Net(1) * Nzt(1)
   Nxi(2) * Net(1) * Nzt(1)
   Nxi(2) * Net(2) * Nzt(1)
   Nxi(1) * Net(2) * Nzt(1)
   Nxi(1) * Net(1) * Nzt(2)
   Nxi(2) * Net(1) * Nzt(2)
   Nxi(2) * Net(2) * Nzt(2)
   Nxi(1) * Net(2) * Nzt(2)
]';

end