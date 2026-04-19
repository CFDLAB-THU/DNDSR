function N = SF_Quad4(xi,et,zt)

Nxi = SF_Line2(xi, 0, 0);
Net = SF_Line2(et, 0, 0);

N = [...
   Nxi(1) * Net(1)
   Nxi(2) * Net(1)
   Nxi(2) * Net(2)
   Nxi(1) * Net(2)
]';

end