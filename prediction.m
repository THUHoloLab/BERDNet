clear;clc;close all;
addpath('.\functions');
%% load trained network
load UNet_5blocks_trained.mat
%% predict hologram
X = imread('.\image_test\object1080_101.bmp');
X = single(X);
dlX = gpuArray(dlarray(X,'SSCB')); 
tic
dlY = forward(dlnet,dlX,'Outputs','tanh','Acceleration','auto');
toc
dlZ = forward(dlnet,dlX);
Y = gather(extractdata(dlY));
Z = gather(extractdata(dlZ));
figure;imshow(imtile({mat2gray(X),mat2gray(Y),mat2gray(Z)},'GridSize',[1,3]),[]);
imwrite(mat2gray(Y),'.\POH_test\POH_101.bmp')