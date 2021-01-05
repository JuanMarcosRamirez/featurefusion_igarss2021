function  out = convolution_2d(int, krnl, Nx, Ny)  

Nz     = size(int, 2);
aux    = reshape(int, Nx, Ny, Nz);
out    = real(ifft2(repmat(krnl, [1 1 Nz]).*fft2(aux)));
out    = reshape(out, Nx*Ny, Nz);
