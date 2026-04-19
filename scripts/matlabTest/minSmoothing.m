niter = 10;
rng(0);



% a = rand(100,1);
a = ones(100,1);
a(1:2) = 1e-10;

%
xs = 1:numel(a);
a0 = a;
for i = 1:niter
a = minSmooth1D(a);
end
plot(xs,a0,xs,a)
%%



function a = minSmooth1D(a)
    aL = circshift(a,1);
    aR = circshift(a,-1);
    aM = (aL + aR + 1 * a)/3 + (aL + aR - 2*a)/2 * 0.5 ;
    a = min(a, aM);




end