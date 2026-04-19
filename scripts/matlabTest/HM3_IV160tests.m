% from commit 60acca36e731abb15f01c841d5f54e586917bbad


close all;


dataHM3.name = "HMLB \alpha = 0.5";
dataHM3.nstep = [5,10,20,40,80]';
dataHM3.err = [7.11196e-04
5.50130e-05
3.57592e-06
2.20320e-07
1.90960e-07
];
dataHM3.cputime = [313.05
417.75
677.51
1109.17
1786.58
];
dataHM3.niter=[62
47
41
35
29
] * 2.1;





dataHM3_V1.name = "HMLB \alpha = 0.55";
dataHM3_V1.nstep = [5,10,20,40,80]';
dataHM3_V1.err = [8.41500e-04
8.48262e-05
8.85605e-06
1.02918e-06
2.40003e-07
];
dataHM3_V1.cputime = [380.35
469.78
714.92
1088.27
1743.31
];
dataHM3_V1.niter = [74
52
42
34
28
] * 2.1;

dataHM3_V2.name = "HMLBM";
dataHM3_V2.nstep = [5,10,20,40,80]';
dataHM3_V2.err = [3.59318e-03
5.78180e-04
7.81526e-05
9.94905e-06
1.27897e-06
];
dataHM3_V2.cputime = dataHM3_V1.cputime./dataHM3_V1.niter.*[84
49
41
36
30
] * 2.1;
dataHM3_V2.niter = [84
49
41
36
30
] * 2.1;

dataESDIRK4.name = "ESDIRK4";
dataESDIRK4.nstep=[5,10,20,40,80]';
dataESDIRK4.err=[4.58830e-04
3.71934e-05
2.33448e-06
1.98742e-07
1.80795e-07
];
dataESDIRK4.cputime=[457.10
706.51
1172.61
2032.98
3215.08
];
dataESDIRK4.niter=[37+38+40+35+43
32+33+35+31+38
27+28+30+26+33
22+23+25+21+28
18+19+21+16+24
    ];

dataBDF2.name = "BDF2";
dataBDF2.nstep = [5,10,20,40,80,160]';
dataBDF2.err = [3.78883e-02
1.57837e-02
5.04585e-03
1.34605e-03
3.42034e-04
8.58484e-05
];
dataBDF2.cputime = [206.05
258.52
380.29
625.75
1006.31
1664.42
];
dataBDF2.niter = [67
47
40
35
30
25
];

datas = {dataHM3, dataHM3_V1, dataHM3_V2, dataESDIRK4, dataBDF2};


for i = 1:numel(datas)
    data = datas{i};

    data.effcputime = 1./data.cputime ./ data.err;
    data.effiter = 1./data.niter./data.nstep./data.err;
    datas{i} = data;
    

end
markers = ["+", "o", "*", ".", "x"];

%% error - nstep
figure(1); clf;
hold on;
for i = 1:numel(datas)
    data = datas{i};

    plot(data.nstep, data.err, 'DisplayName', data.name, "Marker", markers(i), "MarkerSize", 10);


end
xlabel("time steps");
ylabel("\rho error");
xs = linspace(5,160,2);
plot(xs, xs.^(-2), '--k', "DisplayName", '2nd order','LineWidth',1.5);
plot(xs, xs.^(-4), ':k', "DisplayName", '4th order','LineWidth',1.5);

L = legend;
set(gca,'FontName','Times New Roman')
set(gca,"XScale",'log'); set(gca,"YScale", "log");
grid on;

%% error - time
figure(2); clf;
hold on;
for i = 1:numel(datas)
    data = datas{i};

    plot(data.cputime, data.err, 'DisplayName', data.name, "Marker", markers(i), "MarkerSize", 10);


end
set(gca,'FontName','Times New Roman')
xlabel("cputime (s)");
ylabel("\rho error");

L = legend;
set(gca,"XScale",'log'); set(gca,"YScale", "log");
grid on;

%% error - iter
figure(21); clf;
hold on;
for i = 1:numel(datas)
    data = datas{i};

    plot(data.nstep.*data.niter, data.err, 'DisplayName', data.name, "Marker", markers(i), "MarkerSize", 10);


end
set(gca,'FontName','Times New Roman')
xlabel("effective iterations (s)");
ylabel("\rho error");

L = legend;
set(gca,"XScale",'log'); set(gca,"YScale", "log");
grid on;

%% effcputime - nstep
figure(3); clf;
hold on;
for i = 1:numel(datas)
    data = datas{i};

    plot(data.nstep, data.effcputime, 'DisplayName', data.name, "Marker", markers(i), "MarkerSize", 10);


end
set(gca,'FontName','Times New Roman')
xlabel("time steps");
ylabel("efficiency (cpu time)");

L = legend;
set(L, "Location", "northwest");
set(gca,"XScale",'log'); set(gca,"YScale", "log");
grid on;

%% effiter - nstep
figure(31); clf;
hold on;
for i = 1:numel(datas)
    data = datas{i};

    plot(data.nstep, data.effiter, 'DisplayName', data.name, "Marker", markers(i), "MarkerSize", 10);
       

end
set(gca,'FontName','Times New Roman')
xlabel("time steps");
ylabel("efficiency (iteration)");

L = legend;
set(L, "Location", "northwest");
set(gca,"XScale",'log'); set(gca,"YScale", "log");
grid on;

%% niter - time
figure(4); clf;
hold on;
for i = 1:numel(datas)
    data = datas{i};

    plot(data.nstep.*data.niter, data.cputime, 'DisplayName', data.name, "Marker", markers(i), "MarkerSize", 10);
       

end
set(gca,'FontName','Times New Roman')
xlabel("effective iterations");
ylabel("cputime");

L = legend;
set(L, "Location", "northwest");
set(gca,"XScale",'log'); set(gca,"YScale", "log");
grid on;
