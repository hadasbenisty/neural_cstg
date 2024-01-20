% A script to create a nice 1d plot for the performance of the model
number_outputs = 10;
num_meas = 1000;
number_features = number_outputs + 1000;

load('data\paths\paths.mat');
input_creation_handle = @createToyInput;

ii = 1;
SNR = linspace(10, 1, 18);
for ii = 1:length(SNR)
    [Y(ii, :, :), Y_tot(ii, :, :), y(ii, :, :), X(ii, :, :), ...
        X_fake(ii, :, :), X_all(ii, :, :), ...
        features(ii, :, :), N(ii, :, :)] = input_creation_handle(...
    number_outputs, num_meas, SNR(ii), number_features);

    cd cstg_model\
    command = sprintf('%s\\Scripts\\activate.bat && python %s', ...
        venv_py_path, main_py_path);
    [status, cmdout] = system(command);
    cd ..\
    if status == 0
        disp('Script executed successfully');
        disp('Output:');
        disp(cmdout);
    else
        disp('Script execution failed');
    end

    load(fullfile(results_path, 'results.mat'));
    predictions_tot(ii, :, :) = predictions;
    ii = ii +1;
    disp(['SNR: ', num2str(SNR), ' run no.', num2str(ii)]);
end
predictions_tot = permute(predictions_tot, [1, 3, 2]);

%% Calculate the avg error for each iteration
avg_error_percentage = 100 * std(abs(y - predictions_tot), 1, ...
    ndims(predictions_tot)) ./ ...
    std(abs(y),1, ndims(y));
avg_error_percentage = mean(avg_error_percentage, ...
    ndims(avg_error_percentage));

%% plotting the average error as a function of SNR
figure;
scatter(SNR, avg_error_percentage);
title('avg error (SNR)')
ylabel('avg error %')
xlabel('SNR')
