function lowerHalf = getLowerHalf(tensor)
% ---- Description ----
% ---- Inputs ----
% ---- Outputs ----
% Reshape the tensor to a 2D matrix [X*Y, Z]
reshapedMatrix = reshape(tensor, size(tensor,1)*size(tensor,2), []);
lowerHalf = [];
% Process each 'slice' (now each column in reshapedMatrix)
for z = 1:size(tensor,3)
    % Extract the matrix corresponding to the current 'slice'
    currentMatrix = reshape(reshapedMatrix(:, z), size(tensor,1), ...
        size(tensor,2));

    % Extract the lower triangular part, excluding the diagonal
    lowerTriangularMatrix = tril(currentMatrix, -1);

    % Convert to vector and filter out zero elements
    lowerTriangleVector = lowerTriangularMatrix(lowerTriangularMatrix ~= 0);

    % Append it
    lowerHalf(:, z) = lowerTriangleVector;

end