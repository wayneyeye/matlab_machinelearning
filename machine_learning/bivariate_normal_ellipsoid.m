% clear up everything
clear all; close all; clc;
% create a multivariate normal matrix with 2 variables
n=300; % number of entries
mu = [2,3];
sigma = [1,.5;
         .5,2.5];

X = mvnrnd(mu,sigma,n);
%
[V,D]=eig(sigma);
l=diag(D);
l=sqrt(l);
ev1=V(:,1);
ev2=V(:,2);
% visualize a scatter plot

scatter(X(:,1),X(:,2),'.')
axis('equal')
hold on
plot([mu(1),mu(1)+l(1)*ev1(1)],[mu(2),mu(2)+l(1)*ev1(2)])
plot([mu(1),mu(1)+l(2)*ev2(1)],[mu(2),mu(2)+l(2)*ev2(2)])
grid on
angle=acos(ev1'*[1,0]')/pi*180;
%# ellipse centered at (0,0) with axes length
%# (drawn using the default N=36 points)
p = calculateEllipse(mu(1),mu(2), l(1), l(2), -angle);
plot(p(:,1), p(:,2), '-')

prob=[0.5,0.75,0.99];
for p=prob
    c=sqrt(chi2inv(p,2))
    p = calculateEllipse(mu(1),mu(2), c*l(1), c*l(2), -angle);
    plot(p(:,1), p(:,2), '-')
end

function [X,Y] = calculateEllipse(x, y, a, b, angle, steps)
    %# This functions returns points to draw an ellipse
    %#
    %#  @param x     X coordinate
    %#  @param y     Y coordinate
    %#  @param a     Semimajor axis
    %#  @param b     Semiminor axis
    %#  @param angle Angle of the ellipse (in degrees)
    %#
    narginchk(5, 6);
    if nargin<6, steps = 60; end
    beta = -angle * (pi / 180);
    sinbeta = sin(beta);
    cosbeta = cos(beta);
    alpha = linspace(0, 360, steps)' .* (pi / 180);
    sinalpha = sin(alpha);
    cosalpha = cos(alpha);
    X = x + (a * cosalpha * cosbeta - b * sinalpha * sinbeta);
    Y = y + (a * cosalpha * sinbeta + b * sinalpha * cosbeta);
    if nargout==1, X = [X Y]; end
end

