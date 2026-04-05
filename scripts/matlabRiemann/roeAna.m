UL = sym("UL",[5,1], "real");
UR = sym("UR",[5,1], "real");
DU = sym("DU",[5,1], "real");

syms lam1 lam234 lam5

syms vsqr H a gammaM1
velo = sym("velo", [3,1], "real");

ReV = sym(zeros(5));

ReV(1, [1, 2, 5]) = 1;
ReV([3,4],[3,4]) = diag([1,1]);
ReV([2,3,4], [1, 2, 5]) = repmat(velo, [1,3]);
ReV(2, 1) = ReV(2,1) - a;
ReV(2, 5) = ReV(2,5) + a;

ReV(5, 1) = H - velo(1) * a;
ReV(5, 5) = H + velo(1) * a;
ReV(5, 2) = vsqr/2;
ReV(5, [3,4]) = velo([2,3]);

ReV;

alpha = sym("alpha", [5,1]);
alpha([3,4]) = DU([3,4]) - velo([2,3]) * DU(1);
incU4b = DU(5) - dot(alpha([3,4]),velo([2,3]));
alpha(2) = gammaM1 / a^2 * (DU(1) * (H - velo(1)^2) + velo(1) * DU(2) -incU4b);
alpha(1) = (DU(1) * (velo(1) + a) - DU(2) - a * alpha(2)) / (2 * a);
alpha(5) = DU(1) - (alpha(1) + alpha(2));
LeV = jacobian(alpha, DU);


RLamL = ReV * diag([lam1, lam234, lam234, lam234, lam5]) * LeV;

RLamL = simplify(subs(simplify(expand(RLamL)), [velo(2)^2], [vsqr - velo(1)^2 - velo(3)^2]))




