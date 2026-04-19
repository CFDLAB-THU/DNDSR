% from commit bcad3ac73250ad29668826e8855a16c03d6be43c


close all;
iSee = 1; %1: L-1; 2: L-inf


tPerStep = 0.024 * 32;

dataHM3.name = "DITR-U2R2 $c_2 = 0.5$";
dataHM3.nstep = [5,10,20,40]';
dataHM3.err = [1.4334721e-04, 3.8682615e-03
1.1723396e-05, 2.9551381e-04
7.7004339e-07, 1.9710146e-05
5.1462553e-08, 1.2725320e-06
];
dataHM3.niter=[62
38
25
16
] * 2.1;
dataHM3 = calculateCPUTime(dataHM3, tPerStep);




dataHM3_V1.name = "DITR-U2R2 $c_2 = 0.55$";
dataHM3_V1.nstep = [5,10,20,40]';
dataHM3_V1.err = [1.7466948e-04, 4.0513985e-03
1.8347774e-05, 4.5380609e-04
1.9162290e-06, 4.8319092e-05
2.2221195e-07, 5.7932603e-06
];
dataHM3_V1.niter = [72
42
28
17
] * 2.1;
dataHM3_V1 = calculateCPUTime(dataHM3_V1, tPerStep);

dataHM3_V2.name = "DITR-U2R1";
dataHM3_V2.nstep = [5,10,20,40]';
dataHM3_V2.err = [7.9535836e-04, 1.8031397e-02
1.2624319e-04, 3.0072761e-03
1.7724139e-05, 4.2214108e-04
2.2830850e-06, 5.5235486e-05
];
dataHM3_V2.niter = [75
44
29
19
] * 2.1;
dataHM3_V2 = calculateCPUTime(dataHM3_V2, tPerStep);


dataHM3_V3.name = "DITR-U3R1";
dataHM3_V3.nstep = [5,10,20,40]';
dataHM3_V3.err = [ 5.7337465e-04, 1.2583352e-02
5.6508660e-05, 9.8580999e-04
4.6950553e-06, 8.1431576e-05
2.9045962e-07, 4.9338264e-06
];
dataHM3_V3.niter = [75
42
28
17
] * 2.1;
dataHM3_V3 = calculateCPUTime(dataHM3_V3, tPerStep);

dataESDIRK4.name = "ESDIRK4";
dataESDIRK4.nstep=[5,10,20,40]';
dataESDIRK4.err=[9.5912007e-05, 2.3307648e-03
8.2139117e-06, 1.9751468e-04
5.1605488e-07, 1.2730260e-05
4.7479411e-08, 1.1957857e-06
];

dataESDIRK4.niter=[27+28+29+36+31
19+19+20+18+21
15+15+16+14+17
13+13+13+11+14
];
dataESDIRK4 = calculateCPUTime(dataESDIRK4, tPerStep);


dataBDF2.name = "BDF2";
dataBDF2.nstep = [5,10,20,40,80,160]';
dataBDF2.err = [9.0052427e-03, 2.0421172e-01
3.7585756e-03, 8.2666129e-02
1.2162264e-03, 2.4379215e-02
3.3400937e-04, 6.3359691e-03
8.6639193e-05, 1.5903381e-03
2.1912209e-05, 3.9287812e-04
];
dataBDF2.niter = [58
37
24
18
15
13
];
dataBDF2 = calculateCPUTime(dataBDF2, tPerStep);

datas = {dataHM3, dataHM3_V1, dataHM3_V2, dataHM3_V3, dataESDIRK4, dataBDF2};
dataHM3_V2

for i = 1:numel(datas)
    data = datas{i};

    data.effcputime = 1./data.cputime ./ data.err(:,iSee);
    data.effiter = 1./data.niter./data.nstep./data.err(:,iSee);
    datas{i} = data;
    

end
markers = ["o", "x", "d", "s", "x", "+"];

showOrder = 1:100;

%% error - nstep
figure(1); clf;
hold on;
for i = 1:numel(datas)
    data = datas{showOrder(i)};

    plot(2./data.nstep, data.err(:,iSee), 'DisplayName', data.name, "Marker", markers(i), "MarkerSize", 10);


end
setAx(100)
xlabel("$\Delta t$", "Interpreter","latex");
ylabel("$\epsilon_\rho$", "Interpreter","latex");
xs = linspace(2/5,2/50,2);
plot(xs, xs.^(2) * 0.4, '--k', "DisplayName", '2nd order','LineWidth',1.5);
plot(xs, xs.^(3) * 2e-2, '-.k', "DisplayName", '3rd order','LineWidth',1.5);
plot(xs, xs.^(4) * 1e-2, ':k', "DisplayName", '4th order','LineWidth',1.5);

L = leg;
L.Location = "southwest"
set(gca,'FontName','Times New Roman')
set(gca,"XScale",'log'); set(gca,"YScale", "log");
grid on;
print(gcf, "pics/HM3_IV160_fig_1.png", "-dpng", "-r600");
% print(gcf, "pics/HM3_IV160_fig_1.pdf", "-dpdf", "-r600", '-bestfit' );
exportgraphics(gcf,"pics/HM3_IV160_fig_1.pdf",'ContentType','vector','BackgroundColor','none')
%% error - time
figure(2); clf;
hold on;
for i = 1:numel(datas)
    data = datas{showOrder(i)};

    plot(data.cputime, data.err(:,iSee), 'DisplayName', data.name, "Marker", markers(i), "MarkerSize", 10);


end
setAx(200)
xlabel("CPU Time (s)");
ylabel("$\epsilon_\rho$", "Interpreter","latex");
xticks([400,800,1600]);

L = leg;
set(gca,"XScale",'log'); set(gca,"YScale", "log");
grid on;
print(gcf, "pics/HM3_IV160_fig_2.png", "-dpng", "-r600");
% print(gcf, "pics/HM3_IV160_fig_2.pdf", "-dpdf", "-r600", '-bestfit');
exportgraphics(gcf,"pics/HM3_IV160_fig_2.pdf",'ContentType','vector','BackgroundColor','none')
%% error - iter
figure(21); clf;
hold on;
for i = 1:numel(datas)
    data = datas{showOrder(i)};

    plot(data.nstep.*data.niter, data.err(:,iSee), 'DisplayName', data.name, "Marker", markers(i), "MarkerSize", 10);


end
setAx(300)
xlabel("Effective Iterations");
ylabel("$\epsilon_\rho$", "Interpreter","latex");

L = leg;
set(gca,"XScale",'log'); set(gca,"YScale", "log");
grid on;
print(gcf, "pics/HM3_IV160_fig_21.png", "-dpng", "-r600");
% print(gcf, "pics/HM3_IV160_fig_21.pdf", "-dpdf", "-r600", '-bestfit');
exportgraphics(gcf,"pics/HM3_IV160_fig_21.pdf",'ContentType','vector','BackgroundColor','none')
%% effcputime - nstep
figure(3); clf;
hold on;
for i = 1:numel(datas)
    data = datas{showOrder(i)};

    plot(data.nstep, data.effcputime, 'DisplayName', data.name, "Marker", markers(i), "MarkerSize", 10);


end
setAx(400)
xlabel("time steps");
ylabel("efficiency (CPU time)");

L = leg;
set(L, "Location", "northwest");
set(gca,"XScale",'log'); set(gca,"YScale", "log");
grid on;

%% effiter - nstep
figure(31); clf;
hold on;
for i = 1:numel(datas)
    data = datas{showOrder(i)};

    plot(data.nstep, data.effiter, 'DisplayName', data.name, "Marker", markers(i), "MarkerSize", 10);
       

end
setAx(500)
xlabel("time steps");
ylabel("efficiency (iteration)");

L = leg;
set(L, "Location", "northwest");
set(gca,"XScale",'log'); set(gca,"YScale", "log");
grid on;

%% niter - time
figure(4); clf;
hold on;
its = [];
tms= [];
for i = 1:numel(datas)
    data = datas{showOrder(i)};

    plot(data.nstep.*data.niter, data.cputime, 'DisplayName', data.name, "Marker", markers(i), "MarkerSize", 10);
       
its = [its;log10(data.nstep(:).*data.niter(:))];
tms = [tms;log10(data.cputime(:))];
end

xlabel("Effective Iterations");
ylabel("CPU Time (s)");

L = leg;
set(L, "Location", "northwest");
set(gca,"XScale",'log'); set(gca,"YScale", "log");
grid on;
setAx(600)
print(gcf, "pics/HM3_IV160_fig_4.png", "-dpng", "-r600");
% print(gcf, "pics/HM3_IV160_fig_4.pdf", "-dpdf", "-r600", '-bestfit');
exportgraphics(gcf,"pics/HM3_IV160_fig_4.pdf",'ContentType','vector','BackgroundColor','none')

itsnan = isnan(its);
its = its(~itsnan);
tms = tms(~itsnan);
corItTm = corr([its, tms]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function setAx(toff)
posOrig = [0.1100 0.1400];
set(gca,'FontName','Times New Roman', "FontSize", 12, ...
    "Position", [posOrig, 1- posOrig * 2] ,"PositionConstraint", "innerposition")
set(gcf,"PaperOrientation", "landscape", "PaperUnits", "inches", "PaperSize", [6,4.5])
set(gcf,"Position", [[100,100]+toff*[1,0.2],[560, 420]])
end
function L = leg
L = legend("Interpreter","latex");
end

function data = calculateCPUTime(data, timePerIter)
data.cputime = data.niter .* data.nstep  * timePerIter;


end 

