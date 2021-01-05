function [estLabs] = classification(sptFeat, trnLabs, trnIdx)

%% dimensions
[Nx, Ny, Ne]          = size(sptFeat);
n                     = Nx*Ny;

%% adjust data
C                     = reshape(sptFeat, n, Ne);                   

%% classification using SVM
opts                  = templateSVM('KernelFunction','poly',...
                                    'Standardize',1,'KernelScale','auto');
model                 = fitcecoc(C(trnIdx,:), trnLabs,'Learners', opts);
estLabs               = predict(model, C);

end

