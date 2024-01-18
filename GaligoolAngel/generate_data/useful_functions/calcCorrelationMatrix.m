function correlationMatrix = calcCorrelationMatrix(mat)
% ---- Description ----
% The function calculates the correlation matrix for any n-dimensional
% array. It does so by preforming the multiplication along the first two
% dimensions.
% ---- Inputs ----
% mat - a n-dimentional matrix. The first two dimensions are the one we
% want to calculate the correlation for.
% ---- Outputs ----
% correlationMatrix - a size(mat,2) x size(mat,2) x [size(mat, 3:n)] n
% dimentional array, containing a correlation matrix in the first two
% dimensions.
    dims = size(mat);
    if length(dims) > 2
        mat_reshaped = reshape(mat, [dims(1), dims(2), prod(dims(3:end))]);
        mat_reshaped_transposed = permute(mat_reshaped, [2,1,3]);
        correlationMatrix = zeros(size(mat_reshaped,2), ...
            size(mat_reshaped, 2), size(mat_reshaped,3));
        for dim = 1:size(mat_reshaped, 3)
            correlationMatrix(:, :, dim) = ...
                mat_reshaped_transposed(:,:,dim) * mat_reshaped(:,:,dim);
        end
        correlationMatrix = reshape(correlationMatrix, ...
            [dims(2), dims(2), dims(3:end)]);
    else
        correlationMatrix = mat' * mat;
    end
    
end