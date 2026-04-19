function [t,w] = GaussJacobi(n,beta,alpha)
%      [x,w] = GaussJacobi(n,alpha,beta)
%      find Gauss-Jacobi nodes x(1),...x(n) and weights w(1),...,w(n)
%      need alpha>-1, beta>-1
%
%      integral of f(x)*(x+1)^alpha*(1-x)^beta over [-1,1]
%      is approximated by w(1)*f(x(1)) + ... + w(n)*f(x(n))
%
%      integral of f(x)*(x-a)^alpha*(b-x)^beta over [a,b]
%      is approximated by r^(1+alpha+beta)*(w(1)*f(t(1)) + ... + w(n)*f(t(n)))
%      where r=(b-a)/2; t=(a+b)/2+r*x
%
% Example: find integral x^.5*cos(x) from 0 to 3
%   alpha = .5; beta = 0;
%   [x,w] = GaussJacobi(10,alpha,beta); 
%   a=0; b=3; r=(b-a)/2; t=(a+b)/2+r*x;
%   Q = r^(1+alpha+beta)*sum(w.*cos(t))

%        adapted from the more general
%        routine gaussq by Golub and Welsch.  see
%        Golub, G. H., and Welsch, J. H., "Calculation of Gaussian
%        quadrature rules," Mathematics of Computation 23 (April,
%        1969), pp. 221-230.

      nm1 = n - 1;
      ab = alpha + beta;
      abi = 2 + ab;
      muzero = 2^(ab + 1) * gamma(alpha + 1) * gamma(beta + 1) / gamma(abi);
      t(1) = (beta - alpha)/abi;
      b(1) = sqrt(4*(1 + alpha)*(1 + beta)/((abi + 1)*abi*abi));
      a2b2 = beta*beta - alpha*alpha;
      for  i = 2:nm1
         abi = 2*i + ab;
         t(i) = a2b2/((abi - 2)*abi);
         b(i) = sqrt(4*i*(i + alpha)*(i + beta)*(i + ab)/((abi*abi - 1)*abi*abi));
      end
      abi = 2*n + ab;
      if (abi==2)
        t(n) == 0;
      else
        t(n) = a2b2/((abi - 2)*abi);
      end
     w(1) = 1;
     w(2:n) = 0;

     if n>1
      b(n) = 0;
      for l = 1:n
        j = 0;
%     :::::::::: look for small sub-diagonal element ::::::::::
        while 1
         for m = l:n  
            if m==n
              break
            end
            if abs(b(m)) <= eps*(abs(t(m)) + abs(t(m+1)))
              break
            end
         end

         p = t(l);
         if m==l
           break  % while
         end
         if j==30
           warning('gaussj: no convergence after 30 iterations')
           return
         end
         j = j + 1;
%     :::::::::: form shift ::::::::::
         g = (t(l+1) - p) / (2 * b(l));
         r = sqrt(g*g+1);
         g = t(m) - p + b(l) / (g + abs(r)*(2*(g>=0)-1));
         s = 1;
         c = 1;
         p = 0;
         mml = m - l;

%     :::::::::: for i=m-1 step -1 until l do -- ::::::::::
         for ii = 1:mml
            i = m - ii;
            f = s * b(i);
            bb = c * b(i);
            if abs(f) >= abs(g) 
              c = g / f;
              r = sqrt(c*c+1);
              b(i+1) = f * r;
              s = 1 / r;
              c = c * s;
            else
              s = f / g;
              r = sqrt(s*s+1);
              b(i+1) = g * r;
              c = 1 / r;
              s = s * c;
            end
            g = t(i+1) - p;
            r = (t(i) - g) * s + 2 * c * bb;
            p = s * r;
            t(i+1) = g + p;
            g = c * r - bb;
%     :::::::::: form first component of vector ::::::::::
            f = w(i+1);
            w(i+1) = s * w(i) + c * f;
            w(i) = c * w(i) - s * f;
         end

         t(l) = t(l) - p;
         b(l) = g;
         b(m) = 0;
        end    %  while
      end      %  for l = 1:n 
%     :::::::::: order eigenvalues and eigenvectors ::::::::::
      for ii = 2:n
         i = ii - 1;
         k = i;
         p = t(i);

         for j = ii:n
            if t(j) >= p
              continue
            end
            k = j;
            p = t(j);
         end      

         if k==i
           continue
         end
         t(k) = t(i);
         t(i) = p;
         p = w(i);
         w(i) = w(k);
         w(k) = p;
      end    
  end    % if n>1

  w = muzero * w.^2;



