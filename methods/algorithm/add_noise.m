function [noisyData] = add_noise(multImge, SNR, TYPE)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

%% Dimensions
[Nx, Ny, Nz]  = size(multImge);
Nxy           = Nx*Ny;

%% Noise Type
dataMtrx           = reshape(multImge, Nx*Ny, Nz)';
switch TYPE
   case {'WHITE'}
        energy     = norm(dataMtrx(:))^2;                    % energy of the signal
        noisEnrgy  = energy/(10^(SNR/10));            % energy of noise to be added
        varNoise   = noisEnrgy/(length(dataMtrx(:))-1);      % variance of noise to be added
   case {'NON-WHITE'}
        eta        = 0.1;
        variances  = sum(dataMtrx(:).^2)/10^(SNR/10) /Nz/Nxy ;
        quad_term  = exp(-((1:Nz)-Nz/2).^2*eta^2/2);
        varNoise   = variances*Nz*quad_term/sum(quad_term);
    otherwise
        error('parameter is unknown');
end

%% Add Noise
Sig        = diag(varNoise);                   % std. deviation of noise to be added
noise      = sqrtm(Sig)*randn(size(dataMtrx));         % noise   
noisyData  = reshape((dataMtrx + noise)', Nx, Ny, Nz);

end






