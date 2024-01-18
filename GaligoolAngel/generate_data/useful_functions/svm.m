function [crossValAccuracy, testAccuracy, SVMModel, testData, ...
    validationData] = svm(X, Y, k, test)
% SVM - Trains a Support Vector Machine (SVM) classifier, performs k-fold
% cross-validation, and evaluates its performance on a test dataset.
%
% ---- Description ----
% This function takes a dataset, splits it into training and testing sets,
% trains an SVM classifier, performs k-fold cross-validation, and evaluates
% the classifier's accuracy on the testing set.
%
% ---- Inputs ----
% X  : A matrix where each row represents a sample and each column
%      represents a feature.
% Y  : A column vector of labels corresponding to each row/sample in X.
% k  : The number of folds for k-fold cross-validation.
% test: The proportion of the dataset to be held out for testing.
%      This is a value between 0 and 1, where 0.3 would hold out 30% of the
%      data for testing.
%
% ---- Outputs ----
% crossValAccuracy: The accuracy of the SVM model on the cross-validation
%                   data.
% testAccuracy    : The accuracy of the SVM model on the test data.
% SVMModel        : The trained SVM model.

    % Splitting the dataset into training and testing sets
    cv = cvpartition(size(X, 1), 'HoldOut', test); 
    idx = cv.test;

    % Separating into training and test data
    X_train = X(~idx, :);
    Y_train = Y(~idx, :);
    X_test = X(idx, :);
    Y_test = Y(idx, :);
    
    validationData = [X_train, Y_train, find(idx == 0)];
    testData = [X_test, Y_test, find(idx == 1)];

    % Training the SVM classifier
    SVMModel = fitcsvm(X_train, Y_train);

    % Performing k-fold cross-validation
    CVSVMModel = crossval(SVMModel, 'Kfold', k); 

    % Calculating Cross Validation Accuracy
    crossValAccuracy = 1 - kfoldLoss(CVSVMModel, 'LossFun', 'ClassifError');

    % Testing the Model on the test data
    [labels, scores] = predict(SVMModel, X_test);

    % Calculating Test Accuracy
    testAccuracy = sum(Y_test == labels) / length(Y_test);
end
