function [ aff_mat,sigma ] = CalcInitAff2D( d, paramsknn )


nn_dist = sort(d.').';
params.knn = min(paramsknn, size(nn_dist, 2));
sigma = median(reshape(nn_dist(:, 1:paramsknn), size(nn_dist,1)*paramsknn,1));

aff_mat = exp(-d.^2/(2*sigma^2));


