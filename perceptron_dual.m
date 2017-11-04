% clear up everything
clear all; close all; clc;
% create a multivariate normal matrix with 2 variables
% postive
n=100; % number of entries
mu = [2,3];
sigma = [2,-1.5;
         -1.5,3];
X1 = [ones(n,1),mvnrnd(mu,sigma,n)];
Y1=ones(n,1);

% negatives
mu = [4,4];
sigma = [1,-1.5;
         -1.5,5];
X2 = [ones(n,1),mvnrnd(mu,sigma,n)];
Y2=-ones(n,1);
scatter(X1(:,2),X1(:,3),'o')
hold on
grid on
scatter(X2(:,2),X2(:,3),'x')

X=[X1;X2];
Y=[Y1;Y2];

% training -- Dual
sz=size(X);
n=sz(1);
eta=0.1;
macro_n=1000;
% initialize dual variables
alpha=zeros(n,1);

for m=1:macro_n
    eta=eta/m;
    for i= randperm(n) 
    W=X; % n*3 matrix
    Alpha_Y=alpha.*Y; %n*1 matrix
    if Alpha_Y'*W*X(i,:)'*Y(i)<=0
        alpha(i)=eta+alpha(i);
    end   
end
end
W=Alpha_Y'*W
%plot line
x = linspace(-2,8,30);
y = (-W(1)*ones(1,30)-W(2).*x)./W(3);
plot(x,y)


