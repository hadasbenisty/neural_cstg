

% A script to create a toy example for the stg model
real_features_inputs = 2;
fake_features_num = 8;
number_features_overall = real_features_inputs + fake_features_num;
num_meas = 5000;
SNR = 5;
is_order_context = true;
num_of_contexts = 2;
input_creation_handle = @createToyInput;
[y, features, N, context] = ...
    createToyInput(...
    real_features_inputs, num_meas, SNR, number_features_overall, num_of_contexts, is_order_context);

% Calc Min Error Possible
error = mse(y, y - N);
stdN = mean(std(N, 1, 2));
% Save Data
load("..\data\paths\paths.mat");
save(fullfile(relative_path, inputs_path, 'dataset_fake.mat'), 'y', ...
    'features', 'N', 'SNR', 'context');

figure;plot(context);