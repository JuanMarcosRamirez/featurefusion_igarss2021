function [class_map, OA, AA, kappa, class_accuracy] = classify_3Dimage(image, ground_truth, train_indices, test_indices, method)

[Nx, Ny, Nz] = size(image);

image = reshape(image,[Nx*Ny Nz]);

im_trn = image(train_indices,:);
im_tst = image(test_indices,:);

if strcmp(method,'RF')
    nTrees          = 200;    
    RF_classifier   = TreeBagger(nTrees, im_trn, ground_truth(train_indices), 'Method','classification');
    class           = str2double(RF_classifier.predict(im_tst));
elseif strcmp(method,'SVM-PLY')
    t               = templateSVM('KernelFunction','poly','Standardize',1,'KernelScale','auto');
    Mdl             = fitcecoc(im_trn, ground_truth(train_indices),'Learners',t);
    class           = predict(Mdl, im_tst);
end

class_map       = class_map_image(double(ground_truth), class, train_indices, test_indices);
[OA, AA, kappa, class_accuracy] = compute_accuracy(ground_truth(test_indices), class_map(test_indices));