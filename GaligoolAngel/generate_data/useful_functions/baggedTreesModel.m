function [crossValidationAccuracy, testAccuracy, BTModel] = ...
    baggedTreesModel(X, Y, test, numTrees)
% baggedTreesModel - Trains a bagged trees (ensemble of decision trees) model,
% performs out-of-bag error estimation for cross-validation, and evaluates its
% performance on a test dataset.
%
% ---- Description ----
% This function takes a dataset, splits it into training and testing sets,
% trains a bagged trees model using the training set, estimates cross-validation
% accuracy using out-of-bag error, and evaluates the model's accuracy on the
% test set.
%
% ---- Inputs ----
% X        : A matrix where each row represents a sample and each column
%            represents a feature.
% Y        : A column vector of labels corresponding to each row/sample in X.
% test     : The proportion of the dataset to be held out for testing.
%            This should be a value between 0 and 1 (e.g., 0.3 for 30%).
% numTrees : The number of trees to use in the bagged tree ensemble.
%
% ---- Outputs ----
% crossValidationAccuracy: The out-of-bag error estimate, representing the
%                          cross-validation accuracy of the model.
% testAccuracy           : The accuracy of the model on the test data.
% BTModel                : The trained bagged trees model.

    % Splitting the dataset into training and testing sets
    cv = cvpartition(size(X,1), 'HoldOut', test);
    idx = cv.test;

    % Separating into training and test data
    X_train = X(~idx, :);
    Y_train = Y(~idx, :);
    X_test = X(idx, :);
    Y_test = Y(idx, :);

    % Training the Bagged Tree Ensemble
    BTModel = TreeBagger(numTrees, X_train, Y_train, 'OOBPrediction', 'On');

    % Estimating Cross-Validation Accuracy using Out-Of-Bag Error
    crossValidationAccuracy = 1 - oobError(BTModel, 'Mode', 'ensemble');

    % Testing the model on the test data
    [predictions, scores] = predict(BTModel, X_test);

    % Evaluating Test Accuracy
    testAccuracy = sum(str2double(predictions) == Y_test) / length(Y_test);
end
