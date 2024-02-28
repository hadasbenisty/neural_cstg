

% A script to create a toy example for the stg model
real_features_inputs = 2;
fake_features_num = 2;
number_features_overall = real_features_inputs + fake_features_num;
num_meas = 1000;
SNR = 25;
is_order_context = true;
num_of_contexts = 2;
input_creation_handle = @generateToyExample;
[y, features, A, context] = input_creation_handle(num_meas, SNR, real_features_inputs, ...
    fake_features_num, num_of_contexts);
% [y, features, N, context] = ...
%     createToyInput(...
%     real_features_inputs, ...
%     num_meas, SNR, number_features_overall, ...
%     num_of_contexts,  true, false);

% Calc Min Error Possible
% error = mse(y, y - N);
% stdN = mean(std(N, 1, 2));
% Save Data
load("..\data\paths\paths.mat");
save(fullfile(relative_path, inputs_path, 'dataset_fake.mat'), 'y', ...
    'features', 'SNR', 'context');

figure;plot(context);

%% Plot Stuff
fig_example = figure;
scatter3(y(1, [1:500, end-500:end]), y(2, [1:500, end-500:end]),context([1:500, end-500:end]), 10, ...
    context([1:500, end-500:end]), "filled");
colormap('jet');
title("Y as a function of context")
xlabel('feature 1');
ylabel("feature 2");

%% Plot dist 
fig_example_dist = figure;
red = [1, 0, 0];
green = [0, 1, 0];
x = linspace(min(features, [], "all" ) - 1, max(features, [], "all") + 1, 1000);
for feat_i = 1:size(features, 1)
    sigma = std(features(feat_i, :));
    mu= mean(features(feat_i, :));
    if feat_i - real_features_inputs <= 0
        color = green;
    else
        color = red;
    end
    pdf_values = (1 / (sigma * sqrt(2 * pi))) * exp(-((x - mu).^2 / (2 * sigma^2)));
    plot(x, pdf_values, "Color", color);
    hold on;
end
hold off;
title('Gaussian Distributions of the inputs')
