% clear up everything
clear all; close all; clc;
% create a multivariate normal matrix with 2 variables
n=100; % number of entries
mu = [2,3];
sigma = [5,0.5;
         0.5,5];

X = mvnrnd(mu,sigma,n);
Mu_hat=ones(1,n)*X/n;
X_bar=ones(n,1)*Mu_hat;

Sigma_hat=1/n*(X-X_bar)'*(X-X_bar);
disp(Sigma_hat);
mvnorm_loglikelihood(X,[[2,3],reshape(Sigma_hat,1,length(mu)^2)])

% contour plot of likelihood
linsize=10;
x = linspace(-20,50,linsize);
y = linspace(-60,80,linsize);
[X_p,Y_p] = meshgrid(x,y);
sz=size(X_p);
dim=sz(1);
X_pr=reshape(X_p,1,dim^2);
Y_pr=reshape(Y_p,1,dim^2);
Z_r=[];
for i=1:dim^2
   Z_r=[Z_r, mvnorm_loglikelihood(X,[[X_pr(i),Y_pr(i)],reshape(Sigma_hat,1,length(mu)^2)])];
end
Z = reshape(Z_r,dim,dim);
figure
surf(X_p,Y_p,Z)
colormap jet
function [prob]=mvnorm_loglikelihood(X,P)
    shape_X=size(X);
    Mu=P(1:shape_X(2));
    Sigma_v=P(shape_X(2)+1:end);
    Sigma=reshape(Sigma_v,shape_X(2),shape_X(2));
    %disp(Sigma);
    logprob=0;
    for i=1:shape_X(1)
        logpdf=-log((2*pi)^shape_X(2)*sqrt(det(Sigma)))+(-(X(i,:)-Mu)/Sigma*(X(i,:)-Mu)'/2);
        logprob=logprob+logpdf;
    end
    prob=-logprob;
end