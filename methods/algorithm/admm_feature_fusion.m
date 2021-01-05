function Cout = admm_feature_fusion(Yh, Ym, R, Krnl, Ne, lmb1, lmb2, varargin)
% ========================================================
%   Solves the following problem via ADMM:
%
%   History is a structure that contains the objective value, the primal and
%   dual residual norms, and the tolerances for the primal and dual residual
%   norms at each iteration.
%
%   min_{C, X1, X2, X3, Q1, Q2}
%   (1/2)||Phih(CQ1) - Yh||_2^F + Iq(Q1) + ...
%   (lmb1/2)||CQ2 - Ym||_2^F  + Iq(Q2) + \lmb1(||X2||_1 + ||X3||_1)
%   sub. to   
%   |I|     |-I  0  0||X1|   |0|
%   |0|C  + |Hv -I  0||X2| = |0|
%   |0|     |Hh  0 -I||X3|   |0|
%
%   ===== Inputs  ===== 
%   H             : Measurement operator.
%   Y             : Measurement data matrix.
%   lmb           : Regularization parameter
%   varargin      : Structure containing parameters for the solver 
%                   and defined as follows.
%
%                   tol       : Tolerance error.
%                   prnt      : To print dual and residual error
%                   mitr      : Maximum number of iterations.
%                   rho       : Lagragian regularization parameter.
%
%               
%
% 	===== Outputs =============
%   C           : Features
%
% ========================================================

%% set the defaults for the optional parameters
MITR        = 500;      % maximum number of iteration
PRNT        = 1;        % display error
TOL         = 1e-4;
RHO         = 1;        % regularizatio parameter
INIT        = 1;

%% dimensions
[Nx, Ny, Nlz]  = size(Ym);
[Nlx, Nly, Nz] = size(Yh);
Nxy         = Nx*Ny;
Nlxy        = Nlx*Nly;
Ns          = sqrt(Nxy/Nlxy);

% horizontal difference operators
Hh          = zeros(Nx, Ny);
Hh(1,1)     = -1;
Hh(1,end)   = 1;
fHh         = fft2(Hh);
cfHh        = conj(fHh); 

% vertical difference operator
Hv          = zeros(Nx, Ny);
Hv(1,1)     = -1;
Hv(end,1)   = 1;
fHv         = fft2(Hv);
cfHv        = conj(fHv);

%% create operators
fKrnl       = fft2(Krnl);
cfKrnl      = conj(fKrnl);
S           = create_2d_decimator(Nx, Ny, Ns)';
myConv      = @(int, krnl) perform_2d_convolution(int, krnl, Nx, Ny);
Phih        = @(int) S*myConv(int, fKrnl);
Phiht       = @(int) myConv(S'*int, cfKrnl);

                     
%% initialization 
X1          = zeros(Nxy, Ne);
X2          = zeros(Nxy, Ne);
X3          = zeros(Nxy, Ne);
G1          = zeros(Nxy, Ne);
G2          = zeros(Nxy, Ne);
G3          = zeros(Nxy, Ne);

%% save a matrix-vector multiply
Inxy        = speye(Nxy);
Jx          = kron(ones(Ns, 1), speye(Nx/Ns));
Jy          = kron(ones(Ns, 1), speye(Ny/Ns));
J_hat       = kron(Jx, Jy);
D_hat       = (spdiags(cfKrnl(:), 0, Nxy, Nxy)*J_hat)';
d_tmp       = 1./( (lmb1 + RHO)*(Ns^2) + diag(D_hat*D_hat') );  
D_tmp       = spdiags(d_tmp(:), 0, Nlxy, Nlxy);
BD_inv      = Inxy - D_hat'*(D_tmp*D_hat);

%% save a matrix-vector multiply
IL          = 1./( cfHh.*fHh + cfHv.*fHv + 1);
Yh          = reshape(Yh, [], Nz);
Ym          = reshape(Ym, Nxy, []);
PhihtYh     = Phiht(Yh);


%% Set the optional parameters
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'MAX_ITER'
                MITR    = varargin{i+1};
            case 'PRINT'
                PRNT    = varargin{i+1};
            case 'TOLERANCE'
                TOL     = varargin{i+1};
            case 'INITIALIZATION'
                INIT    = varargin{i+1};    
        end
    end
end

%% initialization
if INIT == 1
    [~, ~, Q1]  = svd(Yh, 'eco');
    Q1          = Q1(1:Ne,:);
    Q2          = Q1*R;
else
    Q1          = orth(randn(Ne, Nz)')';
    if Ne > Nlz
        Q2          = orth(rand(Ne, Nlz));
    else
        Q2          = orth(rand(Ne, Nlz)')';
    end
end

%% iterative problem
errTol      = [];    
for t = 1:MITR
        X1old   = X1;
        X2old   = X2;
        X3old   = X3;
          
        % C-update
        Tmp   = PhihtYh*Q1' + lmb1*(Ym*Q2') + RHO*(X1 - G1);
        C     = zeros(Nxy, Ne);
        for ne  = 1:Ne
            Tmp11        = fft2(reshape(Tmp(:,ne), Nx, Ny));
            Tmp22        = ifft2(reshape(BD_inv*Tmp11(:), Nx, Ny));
            C(:,ne)      = real((1/(RHO + lmb1)).*Tmp22(:));
        end
         
        % Q1,Q2-update
        [U1, ~, V1]  = svd(Phih(C)'*Yh, 'eco');
        [U2, ~, V2]  = svd(C'*Ym, 'eco');
        Q1           = U1*V1';
        Q2           = U2*V2';        
        
        % X1-update
        Tmp   = C + G1 + myConv(X2 - G2, cfHh) + myConv(X3 - G3, cfHv);
        X1    = myConv(Tmp, IL); 
        
        % X2,X3-update
        HhX1  = myConv(X1, fHh);
        HvX1  = myConv(X1, fHv);
        %X2    = soft(HhX1 + G2, lmb2/rho);
        %X3    = soft(HvX1 + G3, lmb2/rho);        
        Tmp   = max(sqrt((HhX1 + G2).^2 + (HvX1 + G3).^2)-lmb2/RHO,0);
        X2    = Tmp./(Tmp + lmb2/RHO).*(HhX1 + G2);
        X3    = Tmp./(Tmp + lmb2/RHO).*(HvX1 + G3);
         
        % G-update       
        Tmp1     = C    - X1;
        Tmp2     = HhX1 - X2;
        Tmp3     = HvX1 - X3;
        G1       = G1 + Tmp1;
        G2       = G2 + Tmp2;
        G3       = G3 + Tmp3;       
        
        Cout     = reshape(C, Nx, Ny, Ne);
        
        % compute errors
        ps_norm  = RHO*norm(      (X1 - X1old) + ...
                            myConv(X2 - X2old, cfHh) + ...
                            myConv(X3 - X3old, cfHv), 'fro');
        pr_norm  = sqrt(norm(Tmp1, 'fro')^2 + ...
                        norm(Tmp2, 'fro')^2 + ...
                        norm(Tmp3, 'fro')^2 );


        if PRNT                
            fprintf('%d\t%5.5f\t%5.5f\t%5.5f\t\n', t, ...
                pr_norm, ps_norm, RHO);

            subplot(321); imagesc(reshape(C(:,1), Nx, Ny)) 
            subplot(322); imagesc(reshape(C(:,2), Nx, Ny))  
            subplot(323); imagesc(reshape(C(:,3), Nx, Ny))  
            subplot(324); imagesc(reshape(C(:,4), Nx, Ny)) 
            subplot(325); imagesc(reshape(C(:,5), Nx, Ny))
            err = norm(Yh - Phih(C)*Q1, 'fro')^2 + norm(Ym - C*Q2, 'fro')^2;
            errTol = [errTol err];
            subplot(326); semilogy(errTol)
            drawnow
        end

        if ((abs(pr_norm) < TOL) && (abs(ps_norm) < TOL))
             break;
        end
end
end
