function Map = class_map_image(ground_truth,Y, training_indexes, test_indexes)

Map = zeros(size(ground_truth,1), size(ground_truth,2));
Map(training_indexes) = ground_truth(training_indexes);
Map(test_indexes) = Y;


% for i = 1 : length(training_indexes)
%     Map(training_indexes(i)) = ground_truth(training_indexes(i));
% end
% 
% for i = 1 : length(test_indexes)
%     
% end