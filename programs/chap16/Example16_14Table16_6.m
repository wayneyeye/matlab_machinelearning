% Table 16.6 Example 16.14 : BDF for a simple non-stiff example
% Use dumb fixed point iteration to handle nonlinearity

clear all
format short e
h = [.2,.1,.05,.02,.01,.005,.002]; nh = length(h);

y0 = 1; t0 = 1; tf = 10; ye = 1/tf;
iters = 20;

% BDF(1,1): backward Euler
y(1) = y0; t = t0;
for j = 1:nh
  k = h(j); N = (tf-t0) / k;
  for i = 1:N
    y(i+1) = y(i); % primitive initial guess  
    for ll=1:iters, y(i+1) = y(i) - k*y(i+1)^2; end
  end
  errBDF1(j) = abs(ye - y(N+1));
end
rateBDF1 = log2(errBDF1(2:nh)./errBDF1(1:nh-1)) ./ log2(h(2:nh)./h(1:nh-1));
errBDF1
rateBDF1

% BDF(2,2)
y(1) = y0;
for j = 1:nh
  k = h(j); N = (tf-t0) / k;
  y(2) = 1 /(1+k);
  for i = 2:N 
    y(i+1) = y(i);
    for ll=1:iters, fip1 = -y(i+1)^2; y(i+1) = (4*y(i)-y(i-1))/3 + 2/3*k*fip1; end
  end
  errBDF2(j) = abs(ye - y(N+1));
end
rateBDF2 = log2(errBDF2(2:nh)./errBDF2(1:nh-1)) ./ log2(h(2:nh)./h(1:nh-1));
errBDF2
rateBDF2

% BDF(4,4)
y(1) = y0;
for j = 1:nh 
  k = h(j); N = (tf-t0) / k;
  % extra initial values
  for l=1:3, y(l+1) = 1 / (t0 + k*l); end
  for i = 4:N
    y(i+1) = y(i);
    for ll=1:iters
      fip1 = -y(i+1)^2; 
      y(i+1) = (48*y(i)-36*y(i-1)+16*y(i-2)-3*y(i-3) + 12*k*fip1)/25;
    end
  end
  errBDF4(j) = abs(ye - y(N+1));
end
rateBDF4 = log2(errBDF4(2:nh)./errBDF4(1:nh-1)) ./ log2(h(2:nh)./h(1:nh-1));
errBDF4
rateBDF4