function [Y, Y_tot, y, X, X_fake, X_all, features, N] = createToyInput(...
    number_outputs, num_meas, SNR, number_features) 
    % Create X and real Y
    X = randn([number_outputs, num_meas]);
    [Y, A] = createRandomLinearTransformation(X);
    
    % add noise
    N = mean(abs(Y), 1)/SNR .* randn(size(Y));
    Y_tot = Y + N;
    % Pad X
    X_fake = rand([number_features - number_outputs, num_meas]);
    X_all = [X ; X_fake];
    
    features = X_all;
    y = Y_tot;
end