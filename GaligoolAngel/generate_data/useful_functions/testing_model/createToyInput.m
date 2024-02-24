function [y, features, N, context] = ...
    createToyInput(...
    real_features_inputs, num_meas, ...
    SNR, number_features_overall, num_context, is_order_context, ...
    is_extreme_context)


% Create X and real Y
X = randn([real_features_inputs, num_meas]);
[Y, ~] = createRandomLinearTransformation(X);
% If it is context dependant we need different y's for each context.

sigN = sqrt(var(Y(:))/10^(SNR/10));
% add noise
if isnan(SNR)
    N = zeros(size(Y));
else
    N = sigN* randn(size(Y));
end
Y_tot = Y + N;
% Pad X
fake_features_num = number_features_overall - real_features_inputs;
X_fake = rand(fake_features_num, num_meas);

% Creating Context
switch num_context
    case 1
        z = ones(num_meas ,1);
        X_all = [X_fake;X];
        features = X_all;
    case 2
        z = rand(num_meas ,1);
        if is_order_context
            z = sort(z);
            for context = 1:num_context
                z_fake(z >= (context - 1) / num_context) = context; % When itirating
            % will delete the wrong ones.
            end
            z = z_fake;
        elseif is_extreme_context
                z(X(1, :) > 0) = 1;
                z(X(1, :) <= 0) = 0;
        end
        
        
        % for context = 1:num_context
        %     X_all(:, :, context) = [X_fake; X];
        %     features(:, z == context) = X_all(:, z == context, context);
        % end
        X_all1 = [X_fake;X];
        X_all2 = [X; X_fake];
        features(:, z == 1) = X_all1(:, z == 1);
        features(:, z == 2) = X_all2(:, z == 2);
        
    otherwise
        error('need to code this?')

end




y = Y_tot;
context = z;
end