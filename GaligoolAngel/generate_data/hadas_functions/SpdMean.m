function M = SpdMean(PP, vW)

Symm = @(M) (M + M') / 2;
PP   = cat(3, PP{:});
Np   = size(PP, 3);
if nargin < 2
    vW = ones(Np, 1) / Np;
end

if Np == 1
    M = PP;
    return
end

M  = mean(PP, 3);

% vEps = [];
for ii = 1 : 50
    A = sqrtm(M);
    B = inv(A);
        
    S = zeros(size(M));
    for jj = 1 : Np
        C   = PP(:,:,jj);
        BCB = Symm(B * C * B);
        S   = S + vW(jj) * (A * logm(BCB) * A);
    end
    
    M = Symm(A * expm(Symm(B * S * B)) * A); 
    
    eps = norm(S, 'fro');
    if (eps < 1e-10)
        break;
    end
%     vEps(ii) = eps;
end
% figure; plot(vEps);

end