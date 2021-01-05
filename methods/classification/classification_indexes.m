function [train_indexes, test_indexes, num_samples_train,num_samples_test] = classification_indexes(ground_truth, epsilon, type_param)
% classification_indexes selects randomly a set of training samples from
% the ground truth image and it returns the indexes of both the training
% set and the test set, where epsilon is the ratio of training samples.
%
% [training_indexes, test_indexes, num_samples] =
%                   classification_indexes(ground_truth, epsilon)
%
%   Inputs:
%   ground_truth    = ground truth image
%   epsilon         = training set ratio
%
%   Outputs:
%   train_indexes   = indexes of the training samples
%   test_indexes    = indexes of the test samples
%   num_samples     = number of training samples per class
%
%   Reference:
%
%   [1] Juan Marcos Ramirez and Henry Arguello, "Spectral Image
%   Classification From Data Fusion Compressive Measurements"
%
%   Authors:
%   Juan Marcos Ramirez.
%   Universidad Industrial de Santander, Bucaramanga, Colombia
%   email: juanmarcos26@gmail.com
%
%   Date:
%   May, 2018
%
%   Copyright 2018 Juan Marcos Ramirez Rondon.  [juanmarcos26-at-gmail.com]

%   This program is free software; you can redistribute it and/or modify it
%   under the terms of the GNU General Public License as published by the
%   Free Software Foundation; either version 2 of the License, or (at your
%   option) any later version.
%
%   This program is distributed in the hope that it will be useful, but
%   WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
%   General Public License for more details.
%
%   You should have received a copy of the GNU General Public License along
%   with this program; if not, write to the Free Software Foundation, Inc.,
%   675 Mass Ave, Cambridge, MA 02139, USA.


classes = max(ground_truth(:));
train_indexes = [];
test_indexes = [];

num_samples_train   = zeros(1,classes);
num_samples_test    = zeros(1,classes);

for i=1:classes
    indclass    = find(ground_truth == i);
    len         = length(indclass);
    perm_len    = randperm(len);
    if strcmp(type_param,'percentage')
        num_samples_train(i)    = round(epsilon * len);
    elseif strcmp(type_param,'fixed')
        num_samples_train(i)    = epsilon;
    elseif strcmp(type_param,'vector')
        num_samples_train(i)    = epsilon(i);
    end
    num_samples_test(i)     = length(perm_len(num_samples_train(i) + 1 : end));
    train_indexes   = [train_indexes; indclass(perm_len(1:num_samples_train(i)))];
    test_indexes    = [test_indexes; indclass(perm_len(num_samples_train(i) + 1 : end))];
end