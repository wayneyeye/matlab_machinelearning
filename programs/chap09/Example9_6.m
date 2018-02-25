% Example 9.6 : the value of weak line search
% Try pure Newton first. If no convergence in nmax iterations then 
% redo with the line search strategy
% Uses functions phi.m and funv.m

% exact solution
xe = [.695884386117764,-1.34794219305888]';
phie = phi(xe);
nmax = 20; tol = 1.e-8; alphamax = 1; alphamin = 1.e-6; sigma = 1.e-4;

% first initial guess
x0 = [.75,-1.25]';
% second initial guess
x0 = [0,0.3]';

fprintf ('k      ||x_k - x*||      phi_k - phi*        f_kp_k  \n')

% apply Newton's method and print table row after each step
% bugged!
x = x0;
for k=1:nmax
  [fx,Jx] = feval(@funv,x);
  p = -Jx \ fx;
  fprintf ('%d     %e     %e     %e \n',k-1,norm(x-xe),phi(x)-phie, fx'*p )
  x = x + p;  
  if norm(p) < tol*(1+norm(x)) break, end
end

if k == nmax % pure Newton having failed, retry with line search
  x = x0; alpha = alphamax;
  fprintf ('k      alpha            ||x_k - x*||      phi_k - phi*        f_kp_k  \n')
  for k=1:nmax
    [fx,Jx] = feval(@funv,x);
    p = -Jx \ fx; phix = phi(x);
    fprintf ('%d     %e     %e     %e     %e\n',k-1,alpha,norm(x-xe),phix-phie, fx'*p )  
    pgphi = fx' * p; alpha = alphamax;
    xn = x + alpha * p; phixn = phi(xn);
    while (phixn > phix + sigma * alpha * pgphi) && (alpha > alphamin)
      mu = -0.5 * pgphi * alpha / (phixn - phix - alpha * pgphi );
      if mu < .1 || pgphi >= 0
        mu = .5; % don't trust quadratic interpolation from far away
      end
      alpha = mu * alpha;
      xn = x + alpha * p;
      phixn = phi(xn);
    end     
    x = xn;
    if norm(p) < tol*(1+norm(x)) break, end
  end
end