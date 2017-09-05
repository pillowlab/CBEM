function [f,g,h] = expTransfer(x,g_bar)
scale = 1;
f = exp(min(scale.*x,g_bar));
g = scale.*f;
h = scale.^2.*f;