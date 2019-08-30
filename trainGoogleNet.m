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
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
[imdsTest, ~] = splitEachLabel(testimages,0.1,'randomize');

%numTrainImages = numel(imdsTrain.Labels);
%idx = randperm(numTrainImages,16);
%figure
%for i = 1:16
%    subplot(4,4,i)
%    I = readimage(imdsTrain,idx(i));
%    imshow(I)
%end

%% Load Pretrained Network
net = googlenet;

%Extract the layer graph from the trained network and plot the layer graph.
lgraph = layerGraph(net);
figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
plot(lgraph)

net.Layers(1)
inputSize = net.Layers(1).InputSize;
net.Layers

%Replace Final Layers
lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});

numClasses = numel(categories(imdsTrain.Labels));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);


lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');

figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

%Freeze Initial Layers
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:110) = freezeWeights(layers(1:110));
lgraph = createLgraphUsingConnections(layers,connections);

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
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',20, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false ,...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,lgraph,options);

%% Classify Validation Images
[YPred,scores] = classify(net,augimdsValidation);

%Display sample validation images with their predicted labels.

idx = randperm(numel(imdsValidation.Files),30);
figure
for i = 1:30
    subplot(6,5,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label0 = imdsValidation.Labels(idx(i));
    label = YPred(idx(i));
    label = strcat(string(label0),'-->',string(label),' : ',num2str(100*max(scores(idx(i),:)),3), "%");
    title(string(label));
end


YValidation = imdsValidation.Labels;
accuracy = (mean(YPred == YValidation))*100;
disp(accuracy)

