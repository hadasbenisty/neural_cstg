% Test SNR
real_features_inputs = 4;
fake_features_num = 4;
number_features_overall = real_features_inputs + fake_features_num;
num_meas = 100;
is_order_context = true;
num_of_contexts = 2;

createHandle = @(SNR) createToyInput(real_features_inputs, num_meas, ...
    SNR, number_features_overall, num_of_contexts, is_order_context);
error = []; sigN = [];
for SNR = linspace(100, 0, 100)
    [y, features, N, context] = createHandle(SNR);
    error = [error, sqrt(mse(y, y- N))];
    sigN = [sigN, std(N, 1, 'all')];
end

SNR_error_fig = figure;
plot(linspace(100,0,100), sigN ./ error);
title('Min Error / Error (SNR)')
xlabel('SNR')
ylabel("Min Error / Error")

%% Test Num Meas
% Test SNR
real_features_inputs = 4;
fake_features_num = 4;
number_features_overall = real_features_inputs + fake_features_num;
SNR = 20;
is_order_context = true;
num_of_contexts = 2;

createHandle = @(num_meas) createToyInput(real_features_inputs, num_meas, ...
    SNR, number_features_overall, num_of_contexts, is_order_context);
error = []; sigN = [];
for num_meas = linspace(100, 10^6, 100)
    [y, features, N, context] = createHandle(num_meas);
    error = [error, sqrt(mse(y, y- N))];
    sigN = [sigN, std(N, 1, 'all')];
end

num_meas_error_fig = figure;
plot(linspace(1000, 10^6, 100), sigN ./ error);
title('Min Error / Error (numMeas)')
xlabel('Num Meas')
ylabel("Min Error / Error")