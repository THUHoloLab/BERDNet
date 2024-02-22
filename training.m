clc;clear;close all;
addpath('./functions');
%% network loading
load UNet_5blocks_untrained.mat
%% dataset loading
imdsimage = imageDatastore('./image_train','IncludeSubfolders',true);
imdspoh = imageDatastore('./POH_train','IncludeSubfolders',true);
dsTrain = combine(imdsimage,imdspoh);
dsTrainAug = transform(dsTrain,@(ds) imresizeForImagePaires(ds,[1080 1080]));
miniBatchSize = 4;
mbq = minibatchqueue(dsTrainAug,...
    MiniBatchSize=miniBatchSize,...
    MiniBatchFormat="SSBC");
%% training options
numEpochs = 10;
repeat = 10;
averageGrad = [];
averageSqGrad = [];
learnRate = 1e-3;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;
executionEnvironment = "auto";
[ax1,ax2,lineLossNpcc,lineLossRec1,lineLossRec2,lineLoss]=initializePlots4();
plotFrequency = 10;
%% training

iteration = 0;
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs
    
    reset(mbq);
    
    % Shuffle data.
    shuffle(mbq);

    % Loop over mini-batches.
    while hasdata(mbq)
        
        % Read mini-batch of data.
        [dlX,dlY] = next(mbq); 
  
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
            dlY = gpuArray(dlY);
        end

        for k=1:repeat
         iteration = iteration + 1;
         
        % Evaluate model gradients. 
         [gradients,dlYp,dlZ,lossNpcc,lossRec1,lossRec2,loss] = dlfeval(@modelGradients,dlnet,dlX,dlY);
 
        % Update the network parameters using the Adam optimizer.
        [dlnet,averageGrad,averageSqGrad] = ...
            adamupdate(dlnet,gradients,averageGrad,averageSqGrad,iteration,...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        
        addpoints(lineLossNpcc,iteration,double(gather(extractdata(lossNpcc))))
        addpoints(lineLossRec1,iteration,double(gather(extractdata(lossRec1))))
        addpoints(lineLossRec2,iteration,double(gather(extractdata(lossRec2))))
        addpoints(lineLoss,iteration,double(gather(extractdata(loss))))
  
        % Every plotFequency iterations, plot the training progress.
        if mod(iteration,plotFrequency) == 0            
            % Use the first image of the mini-batch as a validation image.
            dlX1 = dlX(:,:,:,1);
            dlX1 = rescale(dlX1,0,255);
            dlY1 = dlY(:,:,:,1);
            dlYp1 = dlYp(:,:,:,1);
            dlYp1 = rescale(dlYp1,0,255);
            dlZ1 = dlZ(:,:,:,1);
            dlZ1 = rescale(dlZ1,0,255);
            
            % To use the function imshow, convert to uint8.
            Image = mat2gray(uint8(gather(extractdata(dlX1))));
            POH = mat2gray(uint8(gather(extractdata(dlY1))));
            Prediction = mat2gray(uint8(gather(extractdata(dlYp1))));
            Reconstruction = mat2gray(uint8(gather(extractdata(dlZ1))));
            
            % Plot the input image and the output image and increase size
            imshow(imtile({Image,POH,Prediction,Reconstruction},'GridSize', [1 4]),'Parent',ax2);
        end
        
        % Display time elapsed since start of training and training completion percentage.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        title(ax1,"Epoch: " + epoch + ", Iteration: " + iteration +...
            ", LearnRate: "+ learnRate + ", Elapsed: " + string(D))
        drawnow
        
       end
      
    end
    
    learnRate=learnRate*0.9;

end

modelDateTime = string(datetime('now',Format='yyyy-MM-dd-HH-mm-ss'));
save('BERDNet'+modelDateTime+'.mat','dlnet','averageGrad','averageSqGrad');

%% loss function
function [gradients,dlYp,dlZ,lossNpcc,lossRec1,lossRec2,loss] = modelGradients(dlnet,dlX,dlY)
    
    % Calculate the NPCC loss.
    [dlYp] = forward(dlnet,dlX,'Outputs','tanh');
    lossNpcc = npccLoss(dlYp,dlY);
    lossNpcc = (lossNpcc + 1)/2;
    
    % Calculate the reconstruction loss.
    [dlZ] = forward(dlnet,dlX);
    lossRec1 = mseLoss(dlZ,100*dlX);  %100 is the average intensity
    lossRec2 = npccLoss(dlZ,dlX);
    lossRec2 = (lossRec2 + 1)/2;
    lossRec = 0.5*lossRec1+0.5*lossRec2;
    
    % Calculate the total loss.
    loss = 0.5*lossRec+0.5*lossNpcc;
    gradients = dlgradient(loss,dlnet.Learnables);
    
end

function loss = mseLoss(dlA,dlB)
    loss = mean((dlA-dlB).^2,'all');
end

function loss = npccLoss(dlA,dlB)

A = dlA - mean(dlA,[1 2]);
B = dlB - mean(dlB,[1 2]);
A_norm = sqrt(sum(A.^2,[1 2]));
B_norm = sqrt(sum(B.^2,[1 2]));
npcc = -sum(A.*B,[1 2])./(A_norm.*B_norm);
loss = mean(npcc,'all');

end