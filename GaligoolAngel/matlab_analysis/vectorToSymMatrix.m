function symMatrix = vectorToSymMatrix(vec)
    % Determine the size of the matrix
    n = ceil(sqrt(2 * length(vec)));  % Calculate dimension of the matrix

    % Initialize the matrix with NaNs
    symMatrix = zeros(n, n);

    % Fill the lower triangular part of the matrix
    symMatrix(tril(true(n, n), -1)) = vec;
    % Fill the upper triangular part of the matrix
    symMatrix = symMatrix + symMatrix';

    % Fill the diagonal part with NaNs
    symMatrix(symMatrix == 0) = NaN;
end