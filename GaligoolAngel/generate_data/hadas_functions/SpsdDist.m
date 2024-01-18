function [d, dU, dR] = SpsdDist(A, B, r, k)

    Symm = @(M) (M + M') / 2;
    
    if nargin < 4
        k = 1;
    end

    A = Symm(A);
    B = Symm(B);

    [VA, ~] = eigs(A, r);
    [VB, ~] = eigs(B, r);
    
    [OA, S, OB] = svd(VA' * VB);
    vTheta      = acos(diag(S));
    
    UA = VA * OA;
    UB = VB * OB;
    
%     UA = VA;
%     UB = VB * OB * OA';
    
    RA = Symm(UA' * A * UA);
    RB = Symm(UB' * B * UB);
    
    dU = norm(vTheta);
    dR = SpdDist(RA, RB);
    d  = sqrt(dU^2 + k * dR^2);
end