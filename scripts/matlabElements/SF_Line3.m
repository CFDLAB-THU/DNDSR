function N = SF_Line3(xi,et,zt)
N = [...
  -(1 - xi) * xi / 2
  (1 + xi) * xi / 2
  (1 - xi) * (1 + xi);
]';

end