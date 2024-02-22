function [ax1,ax2,lineLossNpcc,lineLossRec1,lineLossRec2,lineLoss]=initializePlots4()   %定义用于训练过程显示的函数

set(0,'defaultfigurecolor','w')

% Initialize training progress plot.
% fig1 = figure;

ax1 = subplot(2,4,1:4);

% Plot the three losses on the same axes.
hold on
lineLossNpcc = animatedline('Color','r');
lineLossRec1 = animatedline('Color','g');
lineLossRec2 = animatedline('Color','y');
lineLoss = animatedline('Color','b');

% Customize appearance of the graph.
legend('NPCC loss','Rec1 loss','Rec2 loss','Total loss','Location','Southwest');
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

% Initialize image plot.
ax2 = subplot(2,4,5:8);
axis off

end