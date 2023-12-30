function dprime = getDprime(X,Y)
labels = unique(Y);
if length(labels) ~= 2
dprime = [];
warning('Dprime is well defined for 2 labels only');
return;
end
meanS = mean(X(:, :, Y==labels(1)), 3);
meanF = mean(X(:, :, Y==labels(2)), 3);

varS = var(X(:, :, Y==labels(1)), [], 3);
varF = var(X(:, :, Y==labels(2)), [], 3);
dprime = (meanS-meanF)./sqrt(0.5*(varS+varF));
dprime = sqrt(sum(dprime.^2));
end
