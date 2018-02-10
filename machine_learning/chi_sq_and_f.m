% clear up everything
clear all; close all; clc;
% chi sq
df=5;
Z=randn(df,1);
sum_sq=[];
for i=1:10000
    Z=randn(df,1);
    sum_sq=[sum_sq ones(1,df)*Z.^2];
end
nbins=100;
histogram(sum_sq,nbins);

% F - dist
df1=2;
df2=1;
ratio=[];
for i=1:10000
    Z1=randn(df1,1);
    sum_sq1=ones(1,df1)*Z1.^2;
    Z2=randn(df2,1);
    sum_sq2=ones(1,df2)*Z2.^2;
    ratio=[ratio sum_sq1/sum_sq2*df2/df1];
end
nbins=100;
histogram(ratio,nbins);

%plot f-dist pdf
X=linspace(0,10);
Y = fpdf(X,5,3);
plot(X,Y);

%plot chi-sq pdf
Y=[]
for df=1:10
    X=linspace(0,10);
    Y = [Y;chi2pdf(X,df)];
end
plot(X,Y);
