syms xi et zt
assume([xi,et,zt], 'real')

P_Nodes = Node_Pyramid5()
SF = @SF_Pyramid5;

NN = size(P_Nodes,1);
DMat = sym(zeros(NN));
D0N = simplify(SF(xi,et,zt));
for i = 1:NN
    DMat(i,:) = SF(P_Nodes(i,1),P_Nodes(i,2),P_Nodes(i,3));
    if(any(isnan(double(DMat)),'all'))
        DMat(i,:) = limit(...
            subs(D0N,[xi,et],[P_Nodes(i,1),P_Nodes(i,2)]),...
            zt, P_Nodes(i,3));
        warning('using zeta''s limit')
    end
end
DMat
assert(logical(norm(DMat - eye(size(DMat))) == 0))


D0N = simplify(SF(xi,et,zt))
D1N = simplify([diff(D0N,xi);diff(D0N,et);diff(D0N,zt)])
% D0NCode = regexprep(fortran(D0N),...
%     ["D0N", "D([+-0])"], ...
%     ["v", "e$1"])
% D1NCode = regexprep(fortran(D1N),...
%     ["D1N", "D([+-0])"], ...
%     ["v", "e$1"])

D0NCode = regexprep(ccode(D0N),...
    ["D0N", "\[(\d+)\]\[(\d+)\]"],...
    ["v","($1,$2)"])
D1NCode = regexprep(ccode(D1N),...
    ["D1N", "\[(\d+)\]\[(\d+)\]"],...
    ["v","($1,$2)"])

syms eta zeta
D0NForm = latex(subs(transpose(D0N),[et,zt],[eta,zeta]))
D1NForm = latex(subs(transpose(D1N),[et,zt],[eta,zeta]))
plot3(double(P_Nodes(:,1)),double(P_Nodes(:,2)),double(P_Nodes(:,3)),'o')

set(gca,'XGrid','on','YGrid','on','ZGrid','off')
xlabel('\xi')
ylabel('\eta')
zlabel('\zeta')
%%
FD0s = matlabFunction(symfun( D0N(3),[xi,et,zt]));
FD1s = matlabFunction(symfun( D1N(3,3),[xi,et,zt]));
[ys,zs] = meshgrid(linspace(-1,1,1001),linspace(0,1,1001));




xiis = 0
D0s = FD0s(xiis, ys(:), zs(:));

D0s_N1 = reshape(D0s,size(ys));
D0s_N1(abs(D0s_N1) > 1.5) = nan;

surf(ys,zs,D0s_N1,'LineStyle','none');
xlabel('\eta');
ylabel('\zeta');
p = patch([-1,1,0],[0,0,1-xiis],[1,1,1],[0,0,0]);
p.FaceColor = 'none';
p.LineWidth = 1;
colorbar;

% axis equal
% view(-16,35)
% zlim([-1,1])

view(0,90)






