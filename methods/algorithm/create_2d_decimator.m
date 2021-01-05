function D    = create_2d_decimator(Nx, Ny, Nd)

tmp1          = zeros(1, Nd);
tmp1(1)       = 1;
tmp2          = zeros(1, Nd);
tmp2(1)       = 1;
ones_nd1      = sparse(tmp1);
ones_nd2      = sparse(tmp2);
Ix            = speye(Nx/Nd);
Iy            = speye(Ny/Nd);
x_dec         = kron(Ix, ones_nd1);
y_dec         = kron(Iy, ones_nd2);
D             = kron(x_dec, y_dec)';  % spatial decimator