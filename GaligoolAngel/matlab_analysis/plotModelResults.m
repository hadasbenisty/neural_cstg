function plotModelResults(y, predictions)
% ---- Description ----
% A function to plot the real results against the results from the model.
% ---- Inputs ----
% y - a matrix, num_of_outputs  x num_of_meas. The real results.
% predictions - a matrix, num_of_outputs x num_of_meas. The model
% predictions. 
% ---- Outputs ----
% None.
    figure;
    title('Model vs. Predictions');
    xlabel('measurement no.')
    x = 1:size(predictions, 2);
    predictions_plot = plot(x, predictions,'o');
    hold on;
    y_plot = scatter(x, y, 10, "filled");
    
    legend([predictions_plot, y_plot], {'Output Predictions', ...
        'Real Output'});
end