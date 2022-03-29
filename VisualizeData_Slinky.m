clear; close all; clc;

TargetIter = 160; % change this parameter to view the training performance

folder = 'data_slinky';
temp = load(strcat('./',folder,'/t_',num2str(TargetIter),'_train.txt'));
SimuSteps = length(temp);

TotalSteps = 1; 
NumCycles = 76;
helixRadius = 0.033; % Slinky radius
TrainTrue = zeros(6,NumCycles,SimuSteps,TotalSteps);
TrainPred = zeros(6,NumCycles,SimuSteps,TotalSteps);
TrainTime = zeros(1,SimuSteps);

temp = load(strcat('./',folder,'/t_',num2str(TargetIter),'_train.txt'));
TrainTime = temp';

for ii = 1:TotalSteps
	temp = load(strcat('./',folder,'/true_',num2str(TargetIter),'_train.txt'));
	TrainTrue(:,:,:,ii) = reshape(temp',[6,size(temp',1)/6,size(temp',2)]);

	temp = load(strcat('./',folder,'/pred_',num2str(TargetIter),'_train.txt'));
	TrainPred(:,:,:,ii) = reshape(temp',[6,size(temp',1)/6,size(temp',2)]);
end

disp('finished data preparation');
pauseTime = 0.001;
h1 = figure(1);
StartStep = max(1,TotalSteps);
StepJump = 1;
SimuJump = 1;
SimuSteps = min(70,SimuSteps); % for training visualizaton
for ii = StartStep:StepJump:TotalSteps
	for jj = 1:SimuJump:SimuSteps
		clf();
		% plot slinky configurations
		hold on; box on;
		for kk = 1:NumCycles
			% plot train ground truth
			% angle
			alphaAngle = TrainTrue(3,kk,jj,ii);
			% top vertex
			topNode = [TrainTrue(1,kk,jj,ii)-helixRadius*sin(alphaAngle),TrainTrue(2,kk,jj,ii)+helixRadius*cos(alphaAngle)];
			% bottom vertex
			bottomNode = [TrainTrue(1,kk,jj,ii)+helixRadius*sin(alphaAngle),TrainTrue(2,kk,jj,ii)-helixRadius*cos(alphaAngle)];
			h_true = plot([topNode(1), bottomNode(1)],[topNode(2), bottomNode(2)],'b','linewidth',1.5);

			% plot train prediction
			% angle
			alphaAngle = TrainPred(3,kk,jj,ii);
			% top vertex
			topNode = [TrainPred(1,kk,jj,ii)-helixRadius*sin(alphaAngle),TrainPred(2,kk,jj,ii)+helixRadius*cos(alphaAngle)];
			% bottom vertex
			bottomNode = [TrainPred(1,kk,jj,ii)+helixRadius*sin(alphaAngle),TrainPred(2,kk,jj,ii)-helixRadius*cos(alphaAngle)];
			h_pred = plot([topNode(1), bottomNode(1)],[topNode(2), bottomNode(2)],'r','linewidth',1.5);
		end
		xlim([-0.15,0.5]);
		ylim([-0.6,0.05]);
		axis equal;
		xlabel('x'); ylabel('y');
		title(strcat('slinky motion:', 32, 'epoch', 32, num2str(TargetIter)));
		legend([h_true, h_pred],'true','prediction','location','southwest');
		% saveas(h1,strcat('./png_slinky/',num2str(jj),'_',num2str(ii),'.png'));
		pause(pauseTime);
	end
	% disp(strcat('finished the', 32, num2str(TargetIter),' epoch'));
end
