function [H] = gaussian_kernel(Nx, Ny, std)

kSize  = ([Nx Ny] - 1)/2;
[X, Y] = meshgrid(-kSize(2):kSize(2),...
                  -kSize(1):kSize(1));

H      = exp(-(X.*X + Y.*Y)/(2*std*std));
H(H<eps*max(H(:))) = 0;

sumh = sum(H(:));
if sumh ~= 0
    H  = H/sumh;
end