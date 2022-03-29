function PlotHist
	clear; close all; clc;
	dataHist = load('./histRecorder.txt');
	figure; set(gcf,'position',[100,100,500,400]); hold on; box on;

	epochs = [1:length(dataHist)];
	yyaxis right;
	plot(epochs,dataHist(:,1),'linewidth',1.5); ylabel('Trajectory length');
	ylim([0,75]); yticks([2 35 70]);
	yyaxis left;
	plot(epochs,dataHist(:,2),'linewidth',1.5); ylabel('Training loss');
	set(gca, 'YScale', 'log');
	xlabel('Epochs'); 
	set(gca,'fontsize',20);
end
