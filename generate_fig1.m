clc
clear
close all

%% add path
addpath(genpath('proposed method'))
addpath(genpath('classification'))

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
parm2.varKrn  = 5;                                                         % standard deviation of gaussian kernel
numFeatures   = [4 5 8 10 12 15 18 20 22 24];                              % number of feature band
parm2.lmb     = 1;                                                         % noise regularization
lambdaTV      = [0.0001,... 
                0.0002,... 
                0.0005,...
                0.0010,...
                0.0020,...
                0.0050,...
                0.0100,...
                0.0200,...
                0.0500,...
                0.1000,... 
                0.2000,... 
                0.5000,... 
                1.0000];                                                   % TV regularization

%% parameaters of classification experirments
nTrn         = 50;                                                         % number of training samples
nRlz         = 5;                                                          % number of realizations

%% Analysis of Parameters
for ii = 1 : length(numFeatures)
    for jj = 1 : length(lambdaTV)
        
        parm2.numFea  = numFeatures(ii);                                   % number of spectral features
        parm2.lmbTV   = lambdaTV(jj);                                      % TV regularization
        moACProp     = zeros(nRlz);
        maACProp     = zeros(nRlz);
        disp(['Number of features: ' num2str(numFeatures(ii))...
            '. lambda_TV: ' num2str(lambdaTV(jj)) '.'])
        for indx1 = 1:nRlz
            % create MS and HS images
            [Ref,...
                noisHsi,...
                noisMsi,...
                Labs]     = create_pavia_dataset(hyperimg, hyperimg_gt,...
                ikonos_sp,...
                parm1);
            
            % estimated features with proposed method
            estFeat    = perform_proposed(noisHsi, noisMsi, parm2);
            
            for indx2 = 1:nRlz
                
                % training samples
                labs         = Labs(:);
                [trnIdx,...
                    tstIdx] = training(labs, nTrn);
                trnLabs      = labs(trnIdx);
                tstLabs      = labs(tstIdx);
                
                % classification using fused features
                estLabs      = classification(estFeat, trnLabs, trnIdx);
                [oAC, aAC]   = results(estLabs(tstIdx), tstLabs);
                maACProp(indx1, indx2)  = aAC;
                moACProp(indx1, indx2)  = oAC;  
            end
        end
        mOA(ii,jj) = mean(moACProp(:));
        mAA(ii,jj) = mean(maACProp(:));
    end
end


hyperRGB = zeros(256,256,3);
hyperRGB(:,:,1) = 0.7*imadjust(mat2gray(hyperimg((1:256) + 100,1:256,64)));
hyperRGB(:,:,2) = imadjust(mat2gray(hyperimg((1:256) + 100,1:256,32)));
hyperRGB(:,:,3) = imadjust(mat2gray(hyperimg((1:256) + 100,1:256,16)));
 
subplot(121)
imshow(hyperRGB);
title('HR image')
subplot(122)
contourf(lambdaTV, numFeatures,100*mOA);
title('Overall accuracy (%)');
xlabel('\lambda_{TV}');
ylabel('N_e');
set(gca, 'XScale', 'log');
colorbar;