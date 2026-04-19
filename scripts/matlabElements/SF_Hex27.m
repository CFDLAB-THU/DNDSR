function N = SF_Hex8(xi,et,zt)

Nxi = SF_Line3(xi, 0, 0);
Net = SF_Line3(et, 0, 0);
Nzt = SF_Line3(zt, 0, 0);

N = [...
   Nxi(1) * Net(1) * Nzt(1) %1
   Nxi(2) * Net(1) * Nzt(1)
   Nxi(2) * Net(2) * Nzt(1)
   Nxi(1) * Net(2) * Nzt(1)
   Nxi(1) * Net(1) * Nzt(2)
   Nxi(2) * Net(1) * Nzt(2)
   Nxi(2) * Net(2) * Nzt(2)
   Nxi(1) * Net(2) * Nzt(2)
   Nxi(3) * Net(1) * Nzt(1) %9
   Nxi(2) * Net(3) * Nzt(1)
   Nxi(3) * Net(2) * Nzt(1)
   Nxi(1) * Net(3) * Nzt(1)
   Nxi(1) * Net(1) * Nzt(3)
   Nxi(2) * Net(1) * Nzt(3)
   Nxi(2) * Net(2) * Nzt(3)
   Nxi(1) * Net(2) * Nzt(3)
   Nxi(3) * Net(1) * Nzt(2)
   Nxi(2) * Net(3) * Nzt(2)
   Nxi(3) * Net(2) * Nzt(2)
   Nxi(1) * Net(3) * Nzt(2)
   Nxi(3) * Net(3) * Nzt(1) %21
   Nxi(3) * Net(1) * Nzt(3)
   Nxi(2) * Net(3) * Nzt(3)
   Nxi(3) * Net(2) * Nzt(3)
   Nxi(1) * Net(3) * Nzt(3)
   Nxi(3) * Net(3) * Nzt(2)
   Nxi(3) * Net(3) * Nzt(3) %27
]';

end