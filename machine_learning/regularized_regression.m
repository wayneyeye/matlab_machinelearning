% clear up everything
clear all; close all; clc;
% create a multivariate normal matrix with 2 variables
n=1000; % number of entries
p=3; %number of variables

% make X and Y
X = [ones(n,1),randn(n,p);];
W= randi(10,p+1,1)
Y=X*W+randn(n,1)*0.5;

% Ridge Regression
W_hat_result=[];
W_hat_closed_result=[];
for lambda=[0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]
    W_0=zeros(1,1+p);
    fun=@(W_hat)Loss_Ridge(W_hat,X,Y,lambda);
    lambda;
    W_hat = fminsearch(fun,W_0);
    W_hat_closed=inv(X'*X+lambda*eye(p+1))*X'*Y;
    W_hat_result=[W_hat_result;W_hat];
    W_hat_closed_result=[W_hat_result;W_hat];
end
W_hat_result
W_hat_closed_result
function [loss]=Loss_Ridge(W_hat,X,Y,lambda)
    Y_hat=X*W_hat';
    R=Y_hat-Y;
    SSR=R'*R;
    loss=SSR+lambda*W_hat*W_hat';
end