function symMatrix = vectorToSymMatrix(vec)
    % Ensure vec is a row vector
    if size(vec, 1) ~= 1 && size(vec, 2) ~= 1
        l = size(vec,2);
    elseif size(vec, 2) == 1
        vec = vec';
        l = size(vec,2);
    end
    
    % Determine the size of the matrix
    n = ceil(sqrt(2 * l));  % Calculate dimension of the matrix
    requiredLength = n * (n + 1) / 2;
    
    % Check if vec needs to be padded with zeros
    if l < requiredLength
        vec = [vec, zeros(1, requiredLength - l)];
    end

    % Initialize the matrix with zeros
    symMatrix = zeros(n, n, size(vec, 1));

    for ii = 1:size(symMatrix, 3)
        % Initialize an n x n matrix for current vector
        symMatrixCurrent = zeros(n, n);
        
        % Calculate indices for the lower triangular part (including diagonal)
        ind = tril(true(n, n));
        
        % Assign vector values to the lower triangular part
        symMatrixCurrent(ind) = vec(ii, 1:requiredLength);
        
        % Mirror the lower triangular part to the upper part
        symMatrixCurrent = symMatrixCurrent + triu(symMatrixCurrent', 1);
        
        % Assign the constructed matrix to the corresponding layer
        symMatrix(:, :, ii) = symMatrixCurrent;
    end
end