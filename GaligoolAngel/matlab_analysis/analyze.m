% loading the data 
load("python_analysis\paths.mat");

load(path_to_results);

load(path_to_data);

if size(y) ~= size(predictions)
    predictions = predictions';
end

% Calculate some metrics
avg_error_percentage = mean(std(abs(Y - predictions), 1, 2) ./ ...
    std(abs(Y),1, 2));

%% Plot Stuff
figure;

x = 1:size(predictions, 2);
predictions_plot = scatter(x, predictions, 10, 'filled');
hold on;
y_plot = scatter(x, Y, 10, "filled");
hold on;
y_noised_plot = scatter(x, Y_tot, 10, 'filled');

title(['Real vs. Predictions SNR = ', num2str(SNR)]);
xlabel('measurement no.');
legend([predictions_plot, y_plot, y_noised_plot], {'Output Predictions', ...
    'Real Output', 'Real Output Noised'});

% Plot the weights 
figure;
x = 1:length(model_weights.hypernetwork_weights);
bar(x, model_weights.hypernetwork_weights);
title('Features Weights')
xlabel('feature number')
