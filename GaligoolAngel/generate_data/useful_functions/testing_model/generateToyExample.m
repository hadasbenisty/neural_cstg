function [Y, X, A, context_labels] = generateToyExample(n_measurements, SNR, n_real_features, n_fake_features, n_contexts)
    % Parameters
    A = randn(n_real_features, n_real_features); % Transformation matrix A

    % Initialize X and Y
    X = zeros(n_measurements, n_real_features + n_fake_features);
    Y = zeros(n_measurements, n_real_features);
    
    X_real = linspace(-1, 0, n_measurements);
    X_real2 = linspace(0, 1, n_measurements);
    X_real = [X_real; X_real2]';

    % Context-specific variation
    context_labels(X_real(:, 1) < -0.5) = 1;
    context_labels(X_real(:, 2) >= 0.5) = 2;

    for context = 1:n_contexts
        context_indices = find(context_labels == context);
        
        % Generate real features for this context
        % Scale and offset to ensure distinctiveness across contexts
        X_real_context = X_real(context_indices, :);
        
        % Generate fake features
        % Keep the scale small and consistent to ensure they're identifiable as fake
        X_fake = randn(length(context_indices), n_fake_features) * 0.1;
        
        % Concatenate real and fake features for X
        X(context_indices, :) = [X_real_context, X_fake];
        
        % Compute Y using Y = AX (only real features contribute)
        Y(context_indices, :) = [~(1-context)* X_real_context(:, context), ...
            ~(2-context) * X_real_context(:, context)]* A + ...
            randn(length(context_indices), n_real_features) / SNR;

        
    end
    Y = Y'; X = X'; A = A';
end
