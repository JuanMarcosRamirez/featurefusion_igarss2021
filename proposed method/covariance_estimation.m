function [Sighat] = covariance_estimation(noisImg, METHOD)

%% dimensions
[N1, N2, N3] = size(noisImg);
n            = N1*N2;

switch METHOD
    case {'MLR'}
        Sighat    = multiple_linear_regression(reshape(noisImg, n, N3)');
    case {'MAD'}  
        Sighat    = zeros(N3);
        for i = 1:N3
           stdhat      = median_absolute_deviation(noisImg(:,:,i));
           Sighat(i,i) = stdhat^2;  
        end
    otherwise
        error('unknown method');
end

end


function [Sighat] = multiple_linear_regression(Y)

%% dimensions
[N3, n]  = size(Y);

%% save computation
small    = 1e-6;
YYt      = Y*Y'; 
iYYt     = inv(YYt + small*eye(N3)); 

%% noise estimation algorithm
H        = zeros(N3, n);  
for i = 1:N3
    XX       = iYYt - (iYYt(:,i)*iYYt(i,:))/iYYt(i,i);
    XXa      = YYt(:,i);
    XXa(i)   = 0; 
    beta     = XX * XXa; beta(i)=0; 
    H(i,:)   = Y(i,:) - beta'*Y;
end   

%% noise estimation
HHt      = H*H';
ZZt      = (Y - H)*(Y - H)';
Rz       = ZZt./n;
Rh       = HHt./n;
Rh       = Rh + sum(diag(Rz))/N3/10^5*eye(N3);
Sighat   = diag(diag(Rh));
end

function estStd = median_absolute_deviation(noisImg)  
N1 = max(size(noisImg, 1));
N2 = max(size(noisImg, 2));
if((N1 >1) && (N2 > 1))
	noisImg = noisImg(2:N1-1, 2:N2-1);
end
H  = noisImg;
n  = max(size(noisImg));
% noise estimate on the first scale (Haar analysis)
if (size(noisImg,1) ==1)
	H = (H(:,1:n-1) - H(:,2:n))/sqrt(2);
	v = median(abs(H));
elseif (size(noisImg,2) ==1)
	H = (H(1:n-1,:) - H(2:n,:))/sqrt(2);
	v = median(abs(H));
else	
	N1=max(size(noisImg,1));
	N2=max(size(noisImg,2));
	H = (H(1:N1-1,:) - H(2:N1,:))/sqrt(2);
	H = (H(:,1:N2-1) - H(:,2:N2))/sqrt(2);
		
	v = median(median(abs(H)));
end
estStd = v/0.6745;
end
