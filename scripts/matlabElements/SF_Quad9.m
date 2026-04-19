function N = SF_Quad9(xi,et,zt)

Nxi = SF_Line3(xi, 0, 0);
Net = SF_Line3(et, 0, 0);

N = [...
   Nxi(1) * Net(1)
   Nxi(2) * Net(1)
   Nxi(2) * Net(2)
   Nxi(1) * Net(2)
   Nxi(3) * Net(1)
   Nxi(2) * Net(3)
   Nxi(3) * Net(2)
   Nxi(1) * Net(3)
   Nxi(3) * Net(3)
]';

end