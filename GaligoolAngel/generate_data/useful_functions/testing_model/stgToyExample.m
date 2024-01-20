% A script to create a toy example for the stg model
number_outputs = 10;
number_features = number_outputs + 1000;
num_meas = 1000;
SNR = 10;

input_creation_handle = @createToyInput;

[Y, Y_tot, y, X, X_fake, X_all, features, N] = input_creation_handle(...
    number_outputs, num_meas, SNR, number_features);
% Save Data
load("data\paths\paths.mat");
save(fullfile(inputs_path, 'dataset'), 'y', 'features', 'Y_tot', 'Y', 'N', ...
    'SNR');