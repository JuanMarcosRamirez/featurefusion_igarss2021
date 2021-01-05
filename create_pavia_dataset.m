function [mbndImg, noisHsi, noisMsi, Labs] = ...
                 create_pavia_dataset(hyperimg, hyperimg_gt, ikonos_sp, parm)
% ========================================================
%
%   ===== Inputs  =====
%   hyperimg   :   pavia university dataset with dimensions (N1 x N2 x N3).  
%   hyperimg_gt:   pavia university labels with dimensions (N1 x N2).    
%   ikonos_sp  :   spectral response sensor multispectral.
%   parm       :   structure containing parameters for the solver 
%                  and defined as follows.
%
%                  varKrn  : variance of blur kernel.
%                  sptFac  : spatial decimation factor.
%                  typNois : type noise ('WHITE' or 'COLORED')
%                  SNR_h   : noise level hyperspectral.
%                  SNR_m   : noise level multispectral.
%                  
% 	===== Outputs =====
%   noisHsi:       noisy hyperspectral dataset
%   noisMsi:       noisy multispectral dataset
%
% ========================================================

%% adjust pavia dataset for experiments 
mbndImg      = hyperimg((1:256) + 100,1:256,:);
mbndImg_gt   = hyperimg_gt((1:256) + 100,1:256);
nClass       = unique(mbndImg_gt(:));
for index = 1:length(nClass)
    tIndex   = mbndImg_gt(:) == nClass(index);
    mbndImg_gt(tIndex) = index - 1;
end
spcRes    = zeros(103, 4);
for index = 1:4
    x     = ikonos_sp(17:103,1);
    v     = ikonos_sp(17:103,index+2);
    xq    = linspace(430, 860, 103);
    spcRes(:,index) = interp1(x, v, xq);
end

%% all dimensions
[Nx, Ny, Nz] = size(mbndImg);                                              % spectral and spatial dimensions
n            = Nx*Ny;                                                      % number of pixels

%% parameters
varKrn       = parm.varKrn;                                                % blur kernel variance 
sptFac       = parm.sptFac;                                                % spatial decimation factor
SNR_h        = parm.SNR_h;                                                 % hyperspectral noise level
SNR_m        = parm.SNR_m;                                                 % multispectral noise level
typNois      = parm.typNois;                                               % type noise ('WHITE' or 'COLORED')

%% create blur kernel 
Krnl         = fftshift(gaussian_kernel(Nx, Ny, varKrn));
fKrnl        = fft2(Krnl); 

%% create multi-spectral y hyper-spectral             
Hsi          = real(ifft2(fft2(mbndImg).*repmat(fKrnl, [1 1 Nz])));
Hsi          = imresize(Hsi, 1/sptFac);
Msi          = reshape(mbndImg, n, Nz);
Msi          = reshape(Msi*spcRes, Nx, Ny, []);
Labs         = reshape(mbndImg_gt, Nx, Ny);

%% noisy observations
noisHsi      = add_noise(Hsi, SNR_h, typNois);                           % add colored noise
noisMsi      = add_noise(Msi, SNR_m, typNois);                           % add colored noise

end

