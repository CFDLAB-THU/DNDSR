

R = sym("r",[3,3]);
assume(R,'real');
%% rank 2 
a = sym("a",[3,3]);
A = sym("A", [6,1]);
a(1,1) = A(1);
a(2,2) = A(2);
a(3,3) = A(3);
a(1,2) = A(4);
a(2,3) = A(5);
a(1,3) = A(6);


for i = 1:3
    for j = 1:3
        ijs = sort([i,j]);
        a(i,j) = a(ijs(1),ijs(2));
    end
end
result =  R * a * R';
result = [result(1,1),result(2,2),result(3,3), result(1,2),result(2,3),result(1,3)]
