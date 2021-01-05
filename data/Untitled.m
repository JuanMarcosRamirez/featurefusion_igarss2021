clear all
close all
clc

load('../Data Classification/PaviaU.mat')
load('../Data Classification/PaviaU_gt.mat')

cmp = zeros(256, 3);
cmp(2, :) = [192/255 192/255 192/255]; % Asphalt
cmp(3, :) = [0/255 255/255 0/255];     % Meadows
cmp(4, :) = [0/255 255/255 255/255];   % Gravel
cmp(5, :) = [0/255 128/255 0/255];     % Trees
cmp(6, :) = [255/255 0/255 255/255];   % Painted Metal
cmp(7, :) = [165/255 81/255 41/255];   % Bare soil
cmp(8, :) = [128/255 0/255 128/255];   % Bitumen
cmp(9, :) = [255/255 0/255 0/255];     % Self-blocking
cmp(10, :) = [255/255 255/255 0/255];  % Shadows
image(paviaU_gt); colormap(cmp)