% clear up everything
clear all; close all; clc;
% create a n by 10 random matrix 
n=5;
A=randn(50,3)
[UU,LL,VV] = svd(A)

