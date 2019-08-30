clc;
close all;


%% LOAD DATA

imds = imageDatastore('Data30', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

testimages = imageDatastore('Data30test', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%Divide the data into training and validation data sets. 
%Use 70% of the images for training and 30% for validation. 
%splitEachLabel splits the images datastore into two new datastores.

[imdsTrain,imdsVal,imdsTest] = splitEachLabel(imds,0.7,0.1,'randomized');

%numTrainImages = numel(imdsTrain.Labels);
%idx = randperm(numTrainImages,16);
%figure
%for i = 1:16
%    subplot(4,4,i)
%    I = readimage(imdsTrain,idx(i));
%    imshow(I)
%end

%% Load Pretrained Network
net = vgg16;
%analyzeNetwork(net)

net.Layers
inputSize = net.Layers(1).InputSize

%% Replace Final Layers
layersTransfer = net.Layers(1:end-3);

numClasses = numel(categories(imdsTrain.Labels))
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%% Train Network

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

%To automatically resize the validation images without performing further data augmentation, 
%use an augmented image datastore without specifying any additional preprocessing operations.

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsVal);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',20, ...
    'MaxEpochs',4, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,lgraph_1.Layers,options);

%% Classify Validation Images
[YPred,scores] = classify(netTransfer,augimdsValidation);

%Display sample validation images with their predicted labels.

idx = randperm(numel(imdsVal.Files),30);
figure
for i = 1:30
    subplot(6,5,i)
    I = readimage(imdsVal,idx(i));
    imshow(I)
    label0 = imdsVal.Labels(idx(i));
    label = YPred(idx(i));
    label = strcat(string(label0),'-->',string(label),' : ',num2str(100*max(scores(idx(i),:)),3), "%");
    title(string(label));
end

YValidation = imdsVal.Labels;
accuracy = (mean(YPred == YValidation))*100;
disp(accuracy)




