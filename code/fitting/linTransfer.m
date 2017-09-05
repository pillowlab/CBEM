function [f,g,h] = linTransfer(x,g_bar)
scale = 1;
f = scale.*x;
g = scale;
h = 0;