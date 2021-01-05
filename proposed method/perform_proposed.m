function [estFeat] = perform_proposed(noisHsi, noisMsi, parm)
% ========================================================
%   ===== Required inputs =====
%   shots:      Number of FPA measurements to be captured  
%   database:   Hyperspectral datacube to be used     
%   tau:        Parameter for sparse reconstruction
%
% 	===== Outputs =====
%   result:     Reconstructed hyperspectral datacube
%   time:       Algorithm execution time
%   psnr:       Peak-Signal-to-Noise_Ratio between original and
%               reconstructed datacubes
% ========================================================

%% parameters
mpRad         = parm.mpRad;
varKrn        = parm.varKrn;
numFea        = parm.numFea;                                               % # of spectral features
lmb           = parm.lmb;                                                  % noisy regularization
lmbTV         = parm.lmbTV;                                                % TV regularization

%% dimensions
[Mx, My, Nz]  = size(noisHsi);                                             % HS image dimensions
[Nx, Ny, Mz]  = size(noisMsi);                                             % MS image dimensions
n             = Nx*Ny;                                                     % total # pixel MS image
m             = Mx*My;                                                     % total # pixel HS image                

%% whitening procedure
Sighat_h      = covariance_estimation(noisHsi, 'MLR');                     % multiple linear regression (MLR)
Sighat_m      = covariance_estimation(noisMsi, 'MAD');                     % median absolute deviation (MAD)
whitHsi       = reshape(noisHsi, m, Nz)*pinv(sqrtm(Sighat_h));
whitMsi       = reshape(noisMsi, n, Mz)*pinv(sqrtm(Sighat_m));
whitHsi       = reshape(whitHsi, Mx, My, Nz);
whitMsi       = reshape(whitMsi, Nx, Ny, Mz);
% whitHsi       = noisHsi;
% whitMsi       = noisMsi;
whitHsi       = whitHsi./max(whitHsi(:));                                  % HS image normalization
whitMsi       = whitMsi./max(whitMsi(:));                                  % MS image normalization

%% morphological profiles in MS image
whitMsiMp     = perform_morphology_profile(whitMsi, mpRad);


%% fusing features using AO-ADMM
estFeat       = ao_admm(whitHsi, whitMsiMp, lmbTV, numFea,...
                                            'LAMBDA',      lmb,...
                                            'VAR_KERNEL',  varKrn,...
                                            'MAXITR_ADMM', 1,...
                                            'MAXITR_AO',   100);
end

