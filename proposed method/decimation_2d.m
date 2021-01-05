function  out = decimation_2d(int, Nd, Nx, Ny)  

Nz     = size(int, 2);
int    = reshape(int, Nx, Ny, Nz);
if Nd > 0  
    Nd     = 1/Nd;
    out    = int(1:Nd:end, 1:Nd:end, :);
else
    Nd     = 1/Nd;
    out    = zeros(Nx, Ny, Nz);
    out(1:Nd:end, 1:Nd:end, :) = int;  
end
