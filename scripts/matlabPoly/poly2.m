syms x y z lx ly lz

dol2D = [0, 0, 0 
1, 0, 0
0, 1, 0
2, 0, 0
1, 1, 0
0, 2, 0
3, 0, 0
2, 1, 0
1, 2, 0
0, 3, 0]';

dol3D = [0, 0, 0
1, 0, 0
0, 1, 0
0, 0, 1
2, 0, 0
0, 2, 0
0, 0, 2
1, 1, 0
0, 1, 1
1, 0, 1
3, 0, 0
0, 3, 0
0, 0, 3
2, 1, 0
1, 2, 0
0, 2, 1
0, 1, 2
1, 0, 2
2, 0, 1
1, 1, 1]';

dFacts= [1, 0, 0, 0
1, 1, 0, 0
1, 2, 2, 0
1, 3, 6, 6];

DIBJ_Poly2 = sym('a', [10,10]);
DIBJ_Poly3 = sym('a', [20,20]);
for i = 1:10
    for j = 1:10
        dx = dol2D(1,i);
        dy = dol2D(2,i);
        dz = dol2D(3,i);
        px = dol2D(1,j);
        py = dol2D(2,j);
        pz = dol2D(3,j);
        DIBJ_Poly2(i,j) = FPolynomial3D(px,py,pz,dx,dy,dz,x,y,z,dFacts) / (lx^dx * ly^dy * lz^dz);
    end
end

for i = 1:20
    for j = 1:20
        dx = dol3D(1,i);
        dy = dol3D(2,i);
        dz = dol3D(3,i);
        px = dol3D(1,j);
        py = dol3D(2,j);
        pz = dol3D(3,j);
        DIBJ_Poly3(i,j) = FPolynomial3D(px,py,pz,dx,dy,dz,x,y,z,dFacts) / (lx^dx * ly^dy * lz^dz);
    end
end

%%
code2 = regexprep(ccode(DIBJ_Poly2(1:3,1:3)),"\[(\d+)\]\[(\d+)\]", "($1,$2)")
code3 = regexprep(ccode(DIBJ_Poly3(1:4,1:4)),"\[(\d+)\]\[(\d+)\]", "($1,$2)")


function v =  FPolynomial3D( px,  py,  pz,  dx,  dy,  dz,  x,  y,  z, dFacts)
    
c = dFacts(px+1,dx+1) * dFacts(py+1,dy+1) * dFacts(pz+1,dz+1);
v = 0;
if c ~= 0
    v =   c * x^(px-dx) * y^(py-dy) * z^(pz-dz) ;
end

end

