function [f, g, h] = logOnePlusExpX(X,maxG)
% soft rectifier + derivatives
%
% maxG = after X>maxG, then f = X
%      if X<-maxG, f = 1e-15

X = X(:);

f = X ;
lessT = X<=-30;
greaterT = X>= maxG;
toFit = ~lessT & ~greaterT;

ex = exp(X(toFit));
f(toFit) = log(1+ex) ;
f(lessT) = 1e-15 ;

if(nargout > 1)
    g = zeros(length(f),1)+1e-15;
    g(greaterT) = 1;
    
    nex = exp(-X(toFit));
    g(toFit) = 1./(1+nex);
    
    if(nargout > 2)
        h = zeros(length(f),1);
        h(toFit) = g(toFit).*(1-g(toFit));
    end
end


end