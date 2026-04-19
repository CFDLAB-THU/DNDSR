clear;
% cfv_staticRecData;
cfv_staticRecData2;

%%
figure(1); clf; cla;
hold on;
plot(1:numel(iter_10x10_GSIteration), iter_10x10_GSIteration / iter_10x10_GSIteration(1),...
    '-o', 'DisplayName', 'GS');
% plot(((1:numel(iter_10x10_GS_GMRES_2))-1) * 4 + 1, iter_10x10_GS_GMRES_2 / iter_10x10_GS_GMRES_2(1), ...
%     '-o', 'DisplayName', 'GS-GMRES 2');
plot(((1:numel(iter_10x10_GS_GMRES_5))-1) * 5 + 1, iter_10x10_GS_GMRES_5 / iter_10x10_GS_GMRES_5(1), ...
    '-o', 'DisplayName', 'GS-GMRES 5');
% plot(((1:numel(iter_10x10_GS_GMRES_10))-1) * 20 + 1, iter_10x10_GS_GMRES_10 / iter_10x10_GS_GMRES_10(1), ...
%     '-o', 'DisplayName', 'GS-GMRES 10');

set(gca,'YScale','log');
L = legend();

%%
isee = 2;
figure(2); clf; cla;
hold on;
errBase = iter_10x10_GSIteration_err(end,isee) - 2e-10;
plotErr(iter_10x10_GSIteration_err, errBase, 1, 0,isee, 'GS');
% plotErr(iter_10x10_GS_GMRES_2_Err, errBase, 4, 1,isee, 'GMRES-BJ 2');
plotErr(iter_10x10_GS_GMRES_5_Err, errBase, 5, 1,isee, 'GMRES-BJ 5');
% plotErr(iter_10x10_GS_GMRES_10_Err, errBase, 20, 1,isee, 'GMRES-BJ 10');
set(gca,'YScale','log');
L = legend();

%%
function plotErr(errdata,errBase, a, b,isee, name)
plot((((1:size(errdata(:,isee),1)) - 1) * a + b)', errdata(:,isee) - errBase,...
    '-o', 'DisplayName', name);
end