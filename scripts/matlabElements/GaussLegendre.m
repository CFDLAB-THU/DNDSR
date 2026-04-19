function [xg,wg] = GaussLegendre(n)
%      [x,w] = GaussLegendre(n)
%      find Gauss-Legendre nodes x(1),...x(n) and weights w(1),...,w(n)
%
%      integral of f(x) over [-1,1]
%      is approximated by w(1)*f(x(1)) + ... + w(n)*f(x(n))
%
%      integral of f(x) over [a,b]
%      is approximated by r*(w(1)*f(t(1)) + ... + w(n)*f(t(n)))
%      where r=(b-a)/2; t=(a+b)/2+r*x
%
% Example: find integral sin(x^2) from 0 to 3
%   [x,w] = GaussLegendre(10); 
%   a=0; b=3; r=(b-a)/2; t=(a+b)/2+r*x;
%   Q = r*sum(w.*sin(t.^2))

eps = 1.5*2^(-53);
m=(n+1)/2;
for i=1:m
  z=cos(pi*(i-0.25)/(n+0.5));
  while 1
    p1=1;
    p2=0;
    for j=1:n
      p3=p2;
      p2=p1;
      p1=((2*j-1)*z*p2-(j-1)*p3)/j;
    end
    pp=n*(z*p1-p2)/(z*z-1);
    z1=z;
    z=z1-p1/pp;
    if abs(z-z1) <= eps
      break
    end
  end
  xg(i) = -z;
  xg(n+1-i) = z;
  wg(i) = 2/((1-z*z)*pp*pp);
  wg(n+1-i)=wg(i);
end