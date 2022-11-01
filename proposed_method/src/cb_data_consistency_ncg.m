function [x,obj] = cb_data_consistency_ncg(A,sino,xg,lambda,niter)
obj = zeros(1,niter);


x0=xg;
delx0 = -( (A'*(A*x0 - sino)) + lambda*(x0-xg) );
alph0 =  normmat2(delx0)/(normmat2(A*delx0) + lambda*normmat2(delx0)) ;
x = x0 + (alph0*delx0);

obj(1) = 0.5*( normmat2(A*x-sino) + (lambda*normmat2(x-xg)) );


s0 = delx0;
for i = 1:niter-1

    delx = -( (A'*(A*x - sino) ) + lambda*(x-xg) );
    
    beta = -normmat2(delx)/(innerp(s0,(delx-delx0))); %Dai-Yuan
    s = delx + (beta*s0);
    
    alph = innerp(s,delx)/(normmat2(A*s) + lambda*normmat2(s));
    x = x + (alph*s);
    
    s0=s;
    delx0 = delx;
    
    obj(i+1) = 0.5*( normmat2(A*x-sino) + (lambda*normmat2(x-xg)) );
end
 
end

function sc = normmat2(X)
sc = sum(X.^2,[1,2,3]);
end

function sc = innerp(X1,X2)
sc = sum(X1.*X2,[1,2,3]);
end