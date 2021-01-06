%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Demo Pavia University
%   
%   Reference:
%
%   [1] Juan Ramirez, Hector Vargas, Jose I, Martinez, and Henry Arguello,
%   Subspace-based Feature Fusion from Hyperspectral and Multispectral
%   Images for Pixel-based Classification.
%
%   Authors:
%   Hector Miguel Vargas Garcia, PhD Student.
%   Universidad Industrial de Santander, Bucaramanga, Colombia
%   email: Hvargas121288@gmail.com 
%   Juan Marcos Ramirez, PhD.
%   Universidad Rey Juan Carlos, Mostoles, Madrid, Spain
%   email: juanmarcos.ramirez@urjc.es juanmarcos26@gmail.com
%
%   Date:
%   January, 2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear
close all

%% add path
addpath(genpath('proposed method'));
addpath(genpath('classification'));
addpath(genpath('methods'));

%% load data
load('data/PaviaU.mat')
load('data/PaviaU_gt.mat')
load('data/PaviaU_color.mat')
load('data/Ikonos_spec_resp.mat')

%% parameters synthetic HS and MS images
parm1         = [];
parm1.varKrn  = 2;                                                         % variance of blur kernel
parm1.sptFac  = 4;                                                         % spatial decimation factor
parm1.typNois = 'COLORED';
parm1.SNR_h   = 30;                                                        % hyperspectral noise level
parm1.SNR_m   = 40;                                                        % multispectral noise level

%% parameters proposed method
parm2         = [];
parm2.mpRad   = [2 4 8 16 32];
parm2.varKrn  = 2;                                                         % standard deviation of gaussian kernel
parm2.numFea  = 20;                                                        % number of spectral features
parm2.lmb     = 1;                                                         % noise regularization
parm2.lmbTV   = 0.01;                                                      % TV regularization

%% Feature fusion stage
[Ref, noisHsi, noisMsi, Labs] = create_pavia_dataset(hyperimg,...
                                                     hyperimg_gt,...
                                                     ikonos_sp,...
                                                     parm1);

% estimated features with proposed method
estFeat    = perform_proposed(noisHsi, noisMsi, parm2);

%% Classification stage
nTrn       = 50;
[train_indices,...
    test_indices,...
    num_samples_train,...
    num_samples_test] = classification_indexes(Labs, nTrn, 'fixed');

classifier = 'SVM-PLY';
[map, OA]  = classify_3Dimage(estFeat,double(Labs), train_indices, test_indices,classifier);

%% Display results

% Building the RGB composite of the MS image
rgbNoiseMS = zeros(size(noisMsi,1),size(noisMsi,2),3);
rgbNoiseMS(:,:,1) = mat2gray(noisMsi(:,:,3));
rgbNoiseMS(:,:,2) = mat2gray(noisMsi(:,:,2));
rgbNoiseMS(:,:,3) = mat2gray(noisMsi(:,:,1));

% Building the RGB composite of the HS image
rgbNoiseHS = zeros(size(noisHsi,1),size(noisHsi,2),3);
rgbNoiseHS(:,:,1) =  0.7*imadjust(mat2gray(noisHsi(:,:,64)));
rgbNoiseHS(:,:,2) = imadjust(mat2gray(noisHsi(:,:,32)));
rgbNoiseHS(:,:,3) = imadjust(mat2gray(noisHsi(:,:,16)));

subplot(221),
imshow(rgbNoiseMS),
title('MS image');
subplot(222),
imshow(rgbNoiseHS),
title('HS image');
subplot(223),
imshow(label2color(Labs,'pavia')),
title('Ground truth');
subplot(224)
imshow(label2color(reshape(map,[256 256]),'pavia')),
title('Classification map'),
xlabel(['OA: ' num2str(OA*100) ' %']);