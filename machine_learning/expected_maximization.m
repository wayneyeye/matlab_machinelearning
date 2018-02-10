% clear up everything
clear all; close all; clc;
% create a multivariate normal matrix with 2 variables
% create a mix of 5 gaussian random variables
n=500; % number of entries
mu = [10,3];
sigma = [5,2;
         2,5];
X = [mvnrnd(mu,sigma,n)];

mu = [6,12];
sigma = [8,-1;
         -1,7];
X = [X;mvnrnd(mu,sigma,n)];

mu = [14,-2];
sigma = [5,2;
         2,9];
X = [X;mvnrnd(mu,sigma,n)];

mu = [-5,-5];
sigma = [6,1;
         1,9];
X = [X;mvnrnd(mu,sigma,n)];

mu = [4,-5];
sigma = [5,-1;
         -1,5];
X = [X;mvnrnd(mu,sigma,n)];

%randomly assign to clusters
m=5; %number of guessed groups
sz=length(X(:,1));
Cluster=randi(m,sz,1);% Initialize Cluster results
PDF_matrix=zeros(sz,m);
max_iter=100;
pt=0.1;%pause time
eta=1; %regularization factor

for iter=1:max_iter
    Cluster_new=zeros(sz,1);
    for i=randperm(m) % Inner circle
       L=(Cluster==i);
       Xi=X(L,:);
       [Mu_hat,Sigma_hat]=MLE_Gaussian(Xi); % Estimation
       Sigma_hat=Sigma_hat+eta*eye(2); % avoid sigularity
       for j=1:sz
          PDF_matrix(j,i)= Gauss_likelihood(Mu_hat,Sigma_hat,X(j,:)); % Update PDF Matrix
       end
    end
    % Maximization
    for j=1:sz
        [val, idx] = max(PDF_matrix(j,:));
        Cluster_new(j)=idx;
    end
    % Stop Criteria
    if Cluster==Cluster_new
        pause(pt);clf;
        c = [0,0,1;1,0,0;0,1,0;0,1,1;1,0,1;];
        for i=1:m         
           L=(Cluster==i);
           Xi=X(L,:);
           %disp(["Cluster",i," ",length(Xi(:,1))]);
           scatter(Xi(:,1),Xi(:,2),10,'MarkerEdgeColor',c(i,:),'MarkerFaceColor',c(i,:),'MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2);
           hold on
        end
        disp("Meet Stopping Criteria");
        disp(["Number of Iterations ", iter]);
        break;
    else
        pause(pt);clf;
        c = [0,0,1;1,0,0;0,1,0;0,1,1;1,0,1;];
        for i=1:m         
           L=(Cluster==i);
           Xi=X(L,:);
           %disp(["Cluster",i," ",length(Xi(:,1))]);
           scatter(Xi(:,1),Xi(:,2),10,'MarkerEdgeColor',c(i,:),'MarkerFaceColor',c(i,:),'MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2);
           hold on
        end
        Cluster=Cluster_new;
    end
    eta=eta/2;
end


function[Mu_hat,Sigma_hat]=MLE_Gaussian(X)
    n=length(X(:,1));
    Mu_hat=ones(1,n)*X/n;
    X_bar=ones(n,1)*Mu_hat;
    Sigma_hat=1/n*(X-X_bar)'*(X-X_bar);
end

function [pdf]=Gauss_likelihood(Mu_hat,Sigma_hat,P)
    Dist=P-Mu_hat;
    pdf=1/((2*pi)^length(Mu_hat)*det(Sigma_hat)^.5);
    pdf=pdf*exp(-.5*Dist*inv(Sigma_hat)*Dist');
end
