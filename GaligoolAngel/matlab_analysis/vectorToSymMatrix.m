function symMatrix = vectorToSymMatrix(vec)
    if size(vec ,1) ~= 1 && size(vec, 2) ~= 1
        l = size(vec,2);
    elseif size(vec, 2) == 1
        vec = vec';
    end
    % Determine the size of the matrix
    n = ceil(sqrt(2 * l));  % Calculate dimension of the matrix

    % Initialize the matrix with NaNs
    symMatrix = zeros(n, n, size(vec, 1));

    % Fill the lower triangular part of the matrix
    symMatrixCurrent = zeros(n, n);
    for ii = 1:size(symMatrix, 3)
        symMatrixCurrent(tril(true(n, n), -1)) = vec(ii, :);
        % Fill the upper triangular part of the matrix
        symMatrixCurrent = symMatrixCurrent + symMatrixCurrent';
    
        % Fill the diagonal part with NaNs
        symMatrixCurrent(symMatrixCurrent == 0) = NaN;
        symMatrix(:, :, ii) = symMatrixCurrent;
    end
end