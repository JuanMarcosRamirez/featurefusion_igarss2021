function outImg = perform_morphology_profile(intImg, radius)

outImg  = [];
npr     = length(radius);
[r,c,p] = size(intImg);
IMcr    = zeros(r,c,npr);
IMor    = IMcr;
for j=1:p
    for i=1:npr
        se2 = strel('disk',radius(i));
        IMcr(:,:,i) = closingbyreconstruction(intImg(:,:,j),se2);
        IMor(:,:,i) = openingbyreconstruction(intImg(:,:,j),se2);
    end
    IMall   = cat(3, IMcr, intImg(:,:,j), IMor);
    outImg  = cat(3, outImg, IMall);
end

end

function [Icbr]=closingbyreconstruction(I,se)
Id = imdilate(I, se);
Icbr = imreconstruct(imcomplement(Id), imcomplement(I));
Icbr = imcomplement(Icbr);
end

function [Iobr]=openingbyreconstruction(I,se)
Ie = imerode(I, se);
Iobr = imreconstruct(Ie, I);
end