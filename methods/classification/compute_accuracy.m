function [oa, aa, kappa, class_acc, confusion] = compute_accuracy(G, G_hat)

confusion = confusionmat(G,G_hat);
oa = trace(confusion) / sum(confusion(:));
class_acc =diag(confusion)./sum(confusion,2);
class_acc(isnan(class_acc))=[];
number=size(class_acc,1);
aa=sum(class_acc)/number;

Pe=(sum(confusion)*sum(confusion,2))/(sum(confusion(:))^2);
kappa = (oa-Pe)/(1-Pe);