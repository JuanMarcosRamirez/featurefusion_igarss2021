function [whitHsi] = hsi_whitening(noisyHsi)

%% Dimensions
[Nx, Ny, Nz] = size(noisyHsi);
Nl           = Nx*Ny;
Yl           = reshape(noisyHsi, Nl, Nz)';

%% Save Computation
small    = 1e-6;
YlYlt    = Yl*Yl'; 
iYlYlt   = inv(YlYlt + small*eye(Nz)); 
Hl       = noise_estimation(iYlYlt, YlYlt, Yl);

%% Noise Estimation
HlHlt    = Hl*Hl';
ZlZlt    = (Yl - Hl)*(Yl - Hl)';
Rzl      = ZlZlt./Nl;
Rhl      = HlHlt./Nl;
Rhl      = Rhl + sum(diag(Rzl))/Nz/10^5*eye(Nz);

%% Noise Covariance
estStd   = diag(Rhl);
whitHsi  = zeros(size(noisyHsi));
for nz = 1:Nz
    whitHsi(:,:,nz) = (1/sqrt(estStd(nz)))*noisyHsi(:,:,nz);
end


function [E] = noise_estimation(iRRt, RRt, Y)
 %% Noise Estimation algorithm
[L, N]  = size(Y);
E       = zeros(L, N);  
for l   = 1:L
    XX       = iRRt - (iRRt(:,l)*iRRt(l,:))/iRRt(l,l);
    XXa      = RRt(:,l);
    XXa(l)   = 0; 
    beta     = XX * XXa; beta(l)=0; 
    E(l,:)   = Y(l,:) - beta'*Y;
end   
