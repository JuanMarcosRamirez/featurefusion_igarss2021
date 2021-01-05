function [whitMsi] = msi_whitening(noisyMsi)

Nz        = size(noisyMsi, 3);
whitMsi   = zeros(size(noisyMsi));
for nz = 1:Nz
    estStd  = noise_estimation(noisyMsi(:,:,nz));
    whitMsi(:,:,nz) = (1/estStd)*noisyMsi(:,:,nz);
end

function estStd = noise_estimation(noisyImg)  
Nx = max(size(noisyImg, 1));
Ny = max(size(noisyImg, 2));
if((Nx >1) && (Ny > 1))
	noisyImg = noisyImg(2:Nx-1, 2:Ny-1);
end
H  = noisyImg;
n  = max(size(noisyImg));
% noise estimate on the first scale (Haar analysis)
if (size(noisyImg,1) ==1)
	H = (H(:,1:n-1) - H(:,2:n))/sqrt(2);
	v = median(abs(H));
elseif (size(noisyImg,2) ==1)
	H = (H(1:n-1,:) - H(2:n,:))/sqrt(2);
	v = median(abs(H));
else	
	Nx=max(size(noisyImg,1));
	Ny=max(size(noisyImg,2));
	H = (H(1:Nx-1,:) - H(2:Nx,:))/sqrt(2);
	H = (H(:,1:Ny-1) - H(:,2:Ny))/sqrt(2);
		
	v = median(median(abs(H)));
end
estStd = v/0.6745;
