clc;
close all;


% Unzip and load the new images as an image datastore. imageDatastore 
% automatically labels the images based on folder names and stores the 
% data as an ImageDatastore object. An image datastore enables you to store 
% large image data, including data that does not fit in memory, and 
% efficiently read batches of images during training of a convolutional 
% neural network.

%unzip('MerchData.zip');
imds = imageDatastore('Data30', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

imdst = imageDatastore('Data30test', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');


% Divide the data into training and validation data sets. 
% Use 70% of the images for training and 30% for validation. splitEachLabel 
% splits the images datastore into two new datastores.

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
%[imdsTest, ~] = splitEachLabel(imdst,0.1,'randomize');

%[imdsValidation,imdsTest]=splitEachLabel(imdsValidation,0.1,'randomize');

% This very small data set now contains 55 training images and 20 
% validation images. Display some sample images.

%numTrainImages = numel(imdsTrain.Labels);
%idx = randperm(numTrainImages,16);
%figure
%for i = 1:16
%    subplot(4,4,i)
%    I = readimage(imdsTrain,idx(i));
%    imshow(I)
%end

% Load the pretrained AlexNet neural network. If Deep Learning Toolbox™
% Model for AlexNet Network is not installed, then the software provides a 
% download link. AlexNet is trained on more than one million images and can 
% classify images into 1000 object categories, such as keyboard, mouse, 
% pencil, and many animals. As a result, the model has learned rich feature 
% representations for a wide range of images.

net = alexnet;


% Use analyzeNetwork to display an interactive visualization of the network 
% architecture and detailed information about the network layers.
%analyzeNetwork(net)

inputSize = net.Layers(1).InputSize

% The last three layers of the pretrained network net are configured for 
% 1000 classes. These three layers must be fine-tuned for the new 
% classification problem. Extract all layers, except the last three, from the
% pretrained network.

layersTransfer = net.Layers(1:end-3);

% Transfer the layers to the new classification task by replacing the last
% three layers with a fully connected layer, a softmax layer, and a 
% classification output layer. Specify the options of the new fully 
% connected layer according to the new data. Set the fully connected layer 
% to have the same size as the number of classes in the new data. To learn 
% faster in the new layers than in the transferred layers, increase the 
% WeightLearnRateFactor and BiasLearnRateFactor values of the fully 
% connected layer.

numClasses = numel(categories(imdsTrain.Labels))

%layers = [
 % layersTransfer
  % fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
   %softmaxLayer
    %classificationLayer];


% The network requires input images of size 227-by-227-by-3, but the images
% in the image datastores have different sizes. Use an augmented image 
% datastore to automatically resize the training images. Specify additional 
% augmentation operations to perform on the training images: randomly flip 
% the training images along the vertical axis, and randomly translate them 
% up to 30 pixels horizontally and vertically. Data augmentation helps 
% prevent the network from overfitting and memorizing the exact details of 
% the training images.

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);


% To automatically resize the validation images without performing further 
% data augmentation, use an augmented image datastore without specifying 
% any additional preprocessing operations.

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdst);

% training specification
options = trainingOptions('sgdm', ...
    'MiniBatchSize',20, ...
    'MaxEpochs',4, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,lgraph_1.Layers,options);


%Calculate the classification accuracy on the validation set. 
%Accuracy is the fraction of labels that the network predicts correctly.
%Classify the validation images using the fine-tuned network.

[YPred,scores] = classify(netTransfer,augimdsValidation);

%Display four sample validation images with their predicted labels.

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
accuracy_Validation= mean(YPred == YValidation)

%% Evaluate Network


validationError = mean(YPred ~= YValidation);
YTrainPred = classify(netTransfer,imdsTrain);
trainError = mean(YTrainPred ~= imdsTrain);
disp("Training error: " + trainError*100 + "%")

%figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
%cm = confusionchart(YValidation,YPred);
%cm.Title = 'Confusion Matrix for Validation Data';
%cm.ColumnSummary = 'column-normalized';
%cm.RowSummary = 'row-normalized';






