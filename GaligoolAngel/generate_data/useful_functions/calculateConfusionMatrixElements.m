function [TP, TN, FP, FN] = calculateConfusionMatrixElements ...
    (predictions, trueValues)
    % This function calculates the elements of the confusion matrix 
    % (TP, TN, FP, FN)
    % for a binary classification problem using vectorized operations.
    % Inputs:
    %   predictions - a vector of predicted values (0 or 1)
    %   trueValues - a vector of true values (0 or 1)
    %
    % Outputs:
    %   TP - True Positives
    %   TN - True Negatives
    %   FP - False Positives
    %   FN - False Negatives

    % Check if the size of predictions and trueValues are the same
    if length(predictions) ~= length(trueValues)
        error('The size of predictions and trueValues must be the same.');
    end

    % Calculate TP, TN, FP, FN using logical indexing
    TP = sum((predictions == 1) & (trueValues == 1));
    TN = sum((predictions == 0) & (trueValues == 0));
    FP = sum((predictions == 1) & (trueValues == 0));
    FN = sum((predictions == 0) & (trueValues == 1));
end
