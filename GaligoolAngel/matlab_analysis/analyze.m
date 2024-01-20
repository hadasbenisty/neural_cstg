% loading the data 
load("data\paths\paths.mat");

load(fullfile(results_path, 'results.mat'));

load(fullfile(inputs_path, 'dataset.mat'));

if size(y) ~= size(predictions)
    predictions = predictions';
end

% Calculate some metrics
avg_error_percentage = 100 * mean(std(abs(y - predictions), 1, 2) ./ ...
    std(abs(y),1, 2));

% Rebuild the Correlation matrix
% CC = vectorToSymMatrix(predictions);
%% Plot Stuff
figure;

x = 1:size(predictions, 2);
predictions_plot = scatter(x, predictions, 10, 'filled');
hold on;
if exist('Y', 'var')
    y_plot = scatter(x, Y, 10, "filled");
else
    y_plot = plot([], []);
end
hold on;
y_noised_plot = scatter(x, y, 10, 'filled');

if exist('SNR', 'var')
    title(['Real vs. Predictions SNR = ', num2str(SNR)]);
else
    title('Real vs. Predictions');
end
xlabel('measurement no.');
legend([predictions_plot, y_plot, y_noised_plot], {'Output Predictions', ...
    'Real Output', 'Real Output Noised'});

% Plot the weights 
figure;
x = 1:length(model_weights.hypernetwork_weights);
bar(x, model_weights.hypernetwork_weights);
title('Features Weights')
xlabel('feature number')

%% Plot the matrix
predicted_features_weights = model_weights.hypernetwork_weights;
corrMat = vectorToSymMatrix(predicted_features_weights);

figure;
imagesc(corrMat);
colorbar;
title('Correlation Weights (0-1)');