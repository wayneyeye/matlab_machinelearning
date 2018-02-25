x0=1.2
f0=sin(x0);
fp=cos(x0)
i=-20:0.5:0;
h=10.^i
err=abs(fp-(sin(x0+h)-f0)./h);
d_err=f0/2*h; %true error using calculus
loglog(h,err,'-*')
hold on 
loglog(h,d_err,'r-.')
xlabel('h')
ylabel('Absolute Error')

