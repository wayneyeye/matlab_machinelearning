% clear up everything
clear all; close all; clc;
% create a multivariate normal matrix with 2 variables
% postive
n=15; % number of entries
mu = [2,1];
sigma = [7,-1;
         -1,6];
X1 = [mvnrnd(mu,sigma,n)];
Y1=ones(n,1);

% negatives
mu = [8,6];
sigma = [7,-3;
         -3,5];
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
%complexity parameter
c=10000;
%theta
theta=3;
%constraints
A=[];
b=[];
Aeq=Y';
beq=0;
lb = zeros(1,n);
ub = c*ones(1,n);
nonlcon=[];
fun = @(alpha) svm_gauss_obj_dual(X,Y,alpha,theta);
opts = optimoptions(@fmincon,'MaxFunctionEvaluations',10000,'Algorithm','sqp','MaxIterations',10000,'FunctionTolerance', 1e-8);
alpha = fmincon(fun,alpha_0,A,b,Aeq,beq,lb,ub,nonlcon,opts);

%plot gaussian_svm classifier
density=100;
x = linspace(-5,15,density);
y = linspace(-8,20,density);
[xx,yy] = meshgrid(x,y);
xx_v=reshape(xx,1,length(xx)^2);
yy_v=reshape(yy,1,length(yy)^2);
zz_v=[];
for i=1:length(xx_v)
    zz_v=[zz_v,gaussian_clf([xx_v(i),yy_v(i)],X,Y,alpha,theta)];
end
zz=reshape(zz_v,length(xx),length(xx));
[C,h] = contourf(xx,yy,zz);
set(h,'LineColor','none');
colormap jet;
hold on;
scatter(X1(:,1),X1(:,2),'o');
scatter(X2(:,1),X2(:,2),'x');

function [dual_margin]=svm_gauss_obj_dual(X,Y,alpha,theta)
    kernel=@(X1,X2)exp(-.5*(X1-X2)*(X1-X2)'/(theta^2));
    dual_margin=0;
    for i=1:length(alpha)
        for j=1:length(alpha)
            dual_margin=dual_margin+kernel(X(i,:),X(j,:))*alpha(i)*alpha(j)*Y(i)*Y(j);
        end
    end
    dual_margin=+.5*dual_margin-alpha*ones(length(alpha),1);
end

function [value]=gaussian_clf(x,X,Y,alpha,theta)
   kernel=@(X1,X2)exp(-.5*(X1-X2)*(X1-X2)'/(theta^2));
   value=0;
   for i=1:length(alpha)
       value=value+kernel(x,X(i,:))*alpha(i)*Y(i);
   end
 
   if value>1
       value=10;
   elseif value<-1
       value=-10;
   else
       value=0;
   end
end


