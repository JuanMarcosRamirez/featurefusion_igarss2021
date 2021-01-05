function Cest = ao_admm(Yh, Ym, lmbTV, Ne, varargin)
% ========================================================
%   Solves the following problem via ADMM:
%
%   History is a structure that contains the objective value, the primal and
%   dual residual norms, and the tolerances for the primal and dual residual
%   norms at each iteration.
%
%   min_{C, X1, X2, X3, Q1, Q2}
%   (1/2)||Phih(CQ1) - Yh||_2^F + Iq(Q1) + ...
%   (lmb/2)||CQ2 - Ym||_2^F  + Iq(Q2) + \lmb_tv(||X2||_1 + ||X3||_1)
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
% 	===== Outputs =============
%   C           : Features
%
% ========================================================

%% set the defaults for the optional parameters
PRINT       = 1;                                                           % display error
MAXITR_ADMM = 100;                                                          % maximum number of iteration of ADMM
MAXITR_AO   = 50;                                                         % maximum number of iteration of AO
TOL_ADMM    = 1e-3; 
TOL_AO      = 1e-4; 
 


var_k       = 1;        % variance kernel blur
rho         = 1;        % lagrange parameter of ADMM
lmb         = 1;       % noise parameter 

%% dimensions
[Nx, Ny, Mz] = size(Ym);
[Mx, My, Nz] = size(Yh);
n           = Nx*Ny;
m           = Mx*My;
Ns          = sqrt(n/m);

%% initialization 
X1          = zeros(n, Ne);
X2          = zeros(n, Ne);
X3          = zeros(n, Ne);
G1          = zeros(n, Ne);
G2          = zeros(n, Ne);
G3          = zeros(n, Ne);

%% Set the optional parameters
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'VAR_KERNEL'
                var_k       = varargin{i+1};
            case 'LAMBDA'
                lmb         = varargin{i+1};
            case 'MAXITR_ADMM'
                MAXITR_ADMM = varargin{i+1};
            case 'MAXITR_AO'
                MAXITR_AO   = varargin{i+1};     
            case 'TOL_AO'
                TOL_AO      = varargin{i+1};   
            case 'PRINT'
                PRINT       = varargin{i+1};    
        end
    end
end

%% create total variation matrices
[fHh, fHv]  = total_variation_matrices(Nx, Ny);
cfHh        = conj(fHh);
cfHv        = conj(fHv);
IL          = 1./( conj(fHh).*fHh + conj(fHv).*fHv + 1);

%% create operators
Krnl        = fftshift(gaussian_kernel(Nx, Ny, var_k));
fKrnl       = fft2(Krnl);
cfKrnl      = conj(fKrnl);
S           = decimator_2d(Nx, Ny, Ns)';
myConv      = @(int, krnl) convolution_2d(int, krnl, Nx, Ny);
Phih        = @(int) S*myConv(int, fKrnl);
Phiht       = @(int) myConv(S'*int, cfKrnl);

%% save a matrix-vector multiply
In          = speye(n);
Jx          = kron(ones(Ns, 1), speye(Nx/Ns));
Jy          = kron(ones(Ns, 1), speye(Nx/Ns));
Jhat        = kron(Jx, Jy);
Shat        = (spdiags(cfKrnl(:), 0, n, n)*Jhat)'; 
dtmp        = 1./( (lmb + rho)*(Ns^2) + diag(Shat*Shat') );  
Dtmp        = spdiags(dtmp(:), 0, m, m);
SBinv       = In - Shat'*(Dtmp*Shat);

%% save a matrix-vector multiply
Yh          = reshape(Yh, m, Nz);
Ym          = reshape(Ym, n, Mz);
PhihtYh     = Phiht(Yh);

%% initialization
[Yh, Qh]    = get_pca_basis(Yh, Ne);
[Ym, Qm]    = get_pca_basis(Ym, Ne);

%% iterative problem
err         = 0;
errRel      = zeros(MAXITR_AO,1);
for t2 = 1:MAXITR_AO

    for t1 = 1:MAXITR_ADMM
        X1old   = X1;

        % C-update
        Tmp     = PhihtYh*Qh' + lmb*(Ym*Qm') + rho*(X1 - G1);
        Tmp11   = reshape(fft2(reshape(Tmp, Nx, Ny, Ne)), n, Ne);
        Tmp22   = ifft2(reshape(SBinv*Tmp11, Nx, Ny, Ne));
        C       = real((1/(rho + lmb)).*reshape(Tmp22, n, Ne));   

        % X1-update
        Tmp   = (C + G1) + myConv(X2 - G2, cfHh) + myConv(X3 - G3, cfHv);
        X1    = myConv(Tmp, IL); 

        % X2,X3-update
        HhX1  = myConv(X1, fHh);
        HvX1  = myConv(X1, fHv);
        aux1  = HhX1 + G2;
        aux2  = HvX1 + G3;
        X2    = zeros(n, Ne);
        X3    = zeros(n, Ne);
        for i = 1:Ne
            Tmp = vector_thresholding_row([aux1(:,i) aux2(:,i)], lmbTV/rho);
            X2(:,i) = Tmp(:,1);
            X3(:,i) = Tmp(:,2);
        end     

        % G-update       
        Tmp1     = C    - X1;
        Tmp2     = HhX1 - X2;
        Tmp3     = HvX1 - X3;
        G1       = G1 + Tmp1;
        G2       = G2 + Tmp2;
        G3       = G3 + Tmp3;       

        Cest     = reshape(C, Nx, Ny, Ne);

        % compute errors of admm
        if mod(t1, 2) == 1
            
            r_norm   = norm([Tmp1;Tmp2;Tmp3],'fro')/...
                       max(norm(C, 'fro'), norm([X1;Tmp2;Tmp3], 'fro'));            
            s_norm   = norm(X1 - X1old, 'fro')/ norm(G1, 'fro');
                                  
                        
            if (PRINT && MAXITR_ADMM~=1)             
                fprintf('%d\t%5.5f\t%5.5f\t%5.5f\t\n',... 
                         t1, r_norm, s_norm, rho);
            end

            if ((r_norm < TOL_ADMM) && (s_norm < TOL_ADMM))
                 break;
            end
        end
    end
    
    % Qm-update
    [U2, ~, V2] = svd(C'*Ym, 'eco');
    Qm          = U2*V2';  
    
    % compute errors of alternating optimization  
    errOld      = err;
    HhC         = myConv(C, fHh);
    HvC         = myConv(C, fHv);
    err         = objective(Yh, Ym, Phih, HhC, HvC,...
                            Qh, Qm, C, lmb, lmbTV);
    errRel(t2)  = abs(err - errOld)/abs(err);
    
    if PRINT                
        fprintf('iteration: %d, relative error: %5.5f\t\n', t2, errRel(t2));
    end

    if errRel(t2) < TOL_AO
        break;
    end
end

end

function err = objective(Yh, Ym, Phih, HhC, HvC, Qh, Qm, C, lmb, lmbTV)
    Ne    = size(C,2);
    errTm = 0;
    for i = 1:Ne
        errTm = errTm + sum(sqrt(HhC(:,i).^2 + HvC(:,i).^2));
    end
    err   = (1/2)*norm(Yh - Phih(C)*Qh, 'fro')^2 +...
            (lmb/2)*norm(Ym - C*Qm, 'fro')^2 + lmbTV*errTm;
end

function outMat = vector_thresholding_row(intMat, tau)

%  perform the vector soft columnwise
N       = size(intMat, 2);
int_nm  = sqrt(sum(intMat.^2, 2));
outMat  = perform_thresholding(int_nm, tau, 'SOFT')./int_nm;
outMat  = repmat(outMat, [1 N]).*intMat;
end

function [intMatm, Qpca] = get_pca_basis(intMat, Nk)

%  perform pca basis estimation
n            = size(intMat, 1);
intMatm      = intMat - repmat(mean(intMat), [n 1]);
[~, ~, Qpca] = svd(intMatm'*intMatm, 'econ');
Qpca         = Qpca(:,1:Nk)';
end

function [fHh, fHv] = total_variation_matrices(Nx, Ny)

% horizontal difference operators
Hh          = zeros(Nx, Ny);
Hh(1,1)     = -1;
Hh(1,end)   = 1;
fHh         = fft2(Hh);

% vertical difference operator
Hv          = zeros(Nx, Ny);
Hv(1,1)     = -1;
Hv(end,1)   = 1;
fHv         = fft2(Hv);

end