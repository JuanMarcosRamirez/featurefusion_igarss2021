clc
clear 
close all

%% add path
addpath(genpath('proposed method'));
addpath(genpath('methods'));

%% file path
load('data/houston_dataset.mat')
rgbImg        = double(rgbImg);
hyprImg       = double(hyprImg);

%% normalize data
hyprImg_nm    = normalize(hyprImg(:), 'range', [0.01 1]);                  % HS image normalization
rgbImg_nm     = normalize(rgbImg(:), 'range', [0.01 1]);                   % MS image normalization
hyprImg_nm    = reshape(hyprImg_nm, size(hyprImg));
rgbImg_nm     = reshape(rgbImg_nm, size(rgbImg));

Nx = size(rgbImg_nm, 1);
Ny = size(rgbImg_nm, 2);
Nz = size(hyprImg_nm, 3);
%% morphological profiles in MS image
mpRad         = [20 50 100 200 500];
rgbImgMp_nm   = perform_morphology_profile(rgbImg_nm, mpRad);


%% fusing features using AO-ADMM                                                     
numFea        = 16;                                                        % number of spectral features
lmbTV         = 0.05;                                                      % TV regularization 
estFeat       = ao_admm(hyprImg_nm, rgbImgMp_nm, lmbTV, numFea,...
                                            'LAMBDA',      1,...           % noise regularization
                                            'VAR_KERNEL',  2,...           % variance of blur kernel
                                            'MAXITR_ADMM', 50,...           % max # iterations ADMM
                                            'MAXITR_AO',   100);           % max # iterations AO
estFeat_nm     = normalize(estFeat(:), 'range', [0.01 1]);
estFeat_nm     = reshape(estFeat_nm, size(estFeat));

%% PCA feature fusion

hyprImg_nm = imresize3(hyprImg_nm,[Nx Ny Nz]);
stckImg_nm = zeros(Nx,Ny,Nz+3);
stckImg_nm(:,:,1:3) = rgbImg_nm;
stckImg_nm(:,:,4:end) = hyprImg_nm;
[~,C_stk] = pca(reshape(stckImg_nm,[Nx*Ny (Nz+3)]));
C_stk = reshape(C_stk(:,1:numFea),[Nx Ny numFea]);

%% Classification stage
nTrn    = 50;                                                              % number of training samples
[train_indices,...
    test_indices,...
    num_samples_train,...
    num_samples_test] = classification_indexes(clasLabs, nTrn, 'fixed');

classifier = 'SVM-PLY';
clasLabs = double(clasLabs);

disp('Classifying HS image');
tic;
[map1, OA1] = classify_3Dimage(imresize3(hyprImg_nm,[Nx Ny Nz]),clasLabs, train_indices, test_indices,classifier);
toc;
disp('Classifying PCA');
tic;
[map2, OA2] = classify_3Dimage(C_stk,clasLabs, train_indices, test_indices,classifier);
toc;
disp('Classifying Proposed');
tic;
[map3, OA3] = classify_3Dimage(estFeat_nm,clasLabs, train_indices, test_indices,classifier);
toc;

%% Display results
rgbhypr = zeros(size(hyprImg,1),size(hyprImg,1),3);
rgbhypr(:,:,1) = imadjust(mat2gray(hyprImg(:,:,24)));
rgbhypr(:,:,2) = imadjust(mat2gray(hyprImg(:,:,14)));
rgbhypr(:,:,3) = imadjust(mat2gray(hyprImg(:,:,8)));

subplot(231),
imshow(mat2gray(rgbImg));
title('RGB image');
subplot(232),
imshow(rgbhypr);
title('HS image');
subplot(233),
imshow(label2color(clasLabs,'houston'));
title('Groud truth');
subplot(234),
imshow(label2color(map1,'houston')),
title('HS'),
xlabel(['OA: ' num2str(OA1*100) ' %']);
subplot(235),
imshow(label2color(map2,'houston')),
title('PCA'),
xlabel(['OA: ' num2str(OA2*100) ' %']);
subplot(236),
imshow(label2color(map3,'houston')),
title('Proposed'),
xlabel(['OA: ' num2str(OA3*100) ' %']);