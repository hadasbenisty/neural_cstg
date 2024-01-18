function [recon_m, proj] = linrecon(x, mu, KernelMat, dim)
x_cent = bsxfun(@minus, x, mu);

proj = x_cent*KernelMat(:,dim);
recon = proj*KernelMat(:,dim)';
recon_m = bsxfun(@plus,recon,mu);

