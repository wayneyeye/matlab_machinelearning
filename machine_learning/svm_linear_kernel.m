% clear up everything
clear all; close all; clc;
% create a multivariate normal matrix with 2 variables
% postive
n=50; % number of entries
mu = [1,2];
sigma = [2,-1;
         -1,3];
X1 = [mvnrnd(mu,sigma,n)];
Y1=ones(n,1);

% negatives
mu = [8,6];
sigma = [1,-1.5;
         -1.5,5];
X2 = [mvnrnd(mu,sigma,n)];
Y2=-ones(n,1);
scatter(X1(:,1),X1(:,2),'o')
hold on
grid on
scatter(X2(:,1),X2(:,2),'x')
X=[X1;X2];
Y=[Y1;Y2];

% initialize vector for alphas
sz=size(X);
n=sz(1);
alpha_0=zeros(1,n);
%constraints
A=[];
b=[];
Aeq=Y';
beq=0;
lb = zeros(1,n);
ub = [];
nonlcon=[];
fun = @(alpha) svm_linear_obj_dual(X,Y,alpha);
opts = optimoptions(@fmincon,'Algorithm','sqp','MaxFunctionEvaluations',5000,'MaxIterations',5000);
alpha = fmincon(fun,alpha_0,A,b,Aeq,beq,lb,ub,nonlcon,opts);

%Get W and t
alpha_y=alpha.*Y';
W=alpha_y*X;
tol=0.01;
for i=1:n
    if alpha(i)>tol
        t=W*X(i,:)'-1/Y(i);
        break;
    end
end

%plot svm_classifier
x=linspace(-5,11);
y=(t*ones(1,length(x))-W(1)*x)./W(2);
y_u=((t+1)*ones(1,length(x))-W(1)*x)./W(2);
y_l=((t-1)*ones(1,length(x))-W(1)*x)./W(2);
y=[y;y_u;y_l];
plot(x,y);


function [dual_margin]=svm_linear_obj_dual(X,Y,alpha)
    left=alpha.*Y';
    gram=X*X';
    right=alpha'.*Y;
    dual_margin=+.5*left*gram*right-alpha*ones(length(alpha),1);
end



