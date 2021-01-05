function [noisImg, Sigma] = add_noise(multImg, SNR, TYPE)
% ========================================================
%   add_noise: add simulated noise in multi channel image
%
%   ===== Inputs  =====
%   multImg:  clean multi channel with dimensions (N1 x N2 x N3)
%   SNR:      signal-to-noise ratio in dB
%   TYPE:     white noise (TYPE = 'WHITE')
%                colored noise (TYPE = 'COLORED')
%
% 	===== Outputs =====
%   noisImg:  noisy multi channel with dimensions (N1 x N2 x N3) 
%   Sigma  :  noise covariance
% ========================================================

%% dimensions
[N1, N2, N3] = size(multImg);
n            = N1*N2;

%% matrix form
Z            = reshape(multImg, n, N3)';

%% noise type
switch TYPE
   case {'WHITE'}
        energy     = norm(Z(:))^2;                                         % energy of the signal
        noisEnrgy  = energy/(10^(SNR/10));                                 % energy of noise to be added
        varNoise   = noisEnrgy/(length(Z(:))-1);                           % variance of noise to be added
   case {'COLORED'}
        eta        = 0.1;
        variances  = sum(Z(:).^2)/10^(SNR/10) /N3/n ;
        quad_term  = exp(-((1:N3)-N3/2).^2*eta^2/2);
        varNoise   = variances*N3*quad_term/sum(quad_term);
    otherwise
        error('parameter is unknown');
end

%% add noise
Sigma      = diag(varNoise);                                               % noise covariance to be added
H          = sqrtm(Sigma)*randn(N3, n);                                    % noise matrix
Y          = (Z + H)';
noisImg    = reshape(Y, N1, N2, N3);

end






