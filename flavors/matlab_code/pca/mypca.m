function [U, mu, eigvals] = mypca(x)
mu= mean(x);
x_cent = bsxfun(@minus, x, mean(x));
[U, p, eigvals] = pca(x_cent);

   