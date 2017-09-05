function [f] = maxabs(xx,m)

s = sign(xx);
s(s==0) = 1;
f = max(abs(xx),abs(m));
f = f.*s;