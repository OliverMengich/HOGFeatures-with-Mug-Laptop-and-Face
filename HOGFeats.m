rootfolder = fullfile('101_ObjectCategories');

categories = {'cup','Faces','chair','laptop'};

imds = imageDatastore(fullfile(rootfolder,categories),'IncludeSubfolders',true,'LabelSource','foldernames');

%%
tbl = countEachLabel(imds);
minsetcount = min(tbl{:,2});
imds= splitEachLabel(imds,minsetcount,'randomize');
[training,test] = splitEachLabel(imds,0.9,'randomize');

%%
im1 = readimage(training,7);
im1 = imresize(im1,[227 227]);
[hog,vis] = extractHOGFeatures(im1,'CellSize',[15 14]);
imshow(im1); hold on; plot(vis)


%%

testimages = numel(training.Files);
trainingFeatures = zeros(testimages,length(hog),'single');

for i=1:testimages
    
   image = readimage(training,i);
   
  % image = rgb2gray(image);
   
   image = imbinarize(image);
   image = imresize(image,[227 227]);
   
   trainingFeatures(i,:) = extractHOGFeatures(image,'CellSize',[15 14]); 
    
    
end

labels = training.Labels;

%%

classifier = fitcecoc(trainingFeatures,labels);
%%
queryimage=readimage(training,204); imshow(queryimage)
queryimage = imresize(queryimage,[227 227]);
queryfeats = extractHOGFeatures(queryimage,'CellSize',[15 14]); 
label = predict(classifier,queryfeats)

%%
testimages = numel(test.Files);
testFeatures = zeros(testimages,length(hog),'single');

for i=1:testimages
    
   image = readimage(test,i);
   
  % image = rgb2gray(image);
   
   image = imbinarize(image);
   image = imresize(image,[227 227]);
   
   testFeatures(i,:) = extractHOGFeatures(image,'CellSize',[15 14]); 
end

testlabels = test.Labels;

%%

predictedlabels = predict(classifier,testFeatures);

%confMat = bsxfun(@rdivide,confMat,sum(confMat,2))
confMat = confusionmat(testlabels,predictedlabels);
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

mean(diag(confMat))

%%
videoplayer = vision.DeployableVideoPlayer();
camera = webcam();

while true
    unk = snapshot(camera);
    queryimage = imresize(unk,[227 227]);
    queryfeats = extractHOGFeatures(queryimage,'CellSize',[15 14]); 
    label = predict(classifier,queryfeats);
    title(label)
    step(videoplayer,unk);
    
end

