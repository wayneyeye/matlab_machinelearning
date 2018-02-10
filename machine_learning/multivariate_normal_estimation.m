% clear up everything
clear all; close all; clc;
% create a multivariate normal matrix with 2 variables
n=100; % number of entries
mu = [2,3];
sigma = [1,1.5;
         1.5,3];

X = mvnrnd(mu,sigma,n);

muhat=ones(1,n)*X/n; % estimated mean vector (row)
% estimated covariance matrix
m_muhat=ones(n,1)*muhat;
sigma_hat=1/(n-1)*(X-m_muhat)'*(X-m_muhat);

% verify unbiased estimators
s=10000;
sum_muhat=zeros(1,2);
sum_sigma_hat=zeros(2,2);
for i = 1:s
    muhat=ones(1,n)*X/n; % estimated mean vector (row)
    m_muhat=ones(n,1)*muhat;
    sigma_hat=1/(n-1)*(X-m_muhat)'*(X-m_muhat);
    sum_muhat=sum_muhat+muhat;
    sum_sigma_hat=sum_sigma_hat+sigma_hat;
end

disp(["muhat ",]);
sum_muhat./s
disp(["sigmahat ",]);
sum_sigma_hat./s

