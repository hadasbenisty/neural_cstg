function [testAccuracy, crossValAccuracy, hyperParam] = trainSVM(X, Y, k, testRatio)
    % Split the dataset into training and testing sets
    cvPartition = cvpartition(Y, 'HoldOut', testRatio);
    XTrain = X(cvPartition.training, :);
    YTrain = Y(cvPartition.training, :);
    XTest = X(cvPartition.test, :);
    YTest = Y(cvPartition.test, :);
    
    % Use a grid search to find the best hyperparameters. Here, we're just using a simple example
    % of varying the box constraint. You can add more hyperparameters like kernel scale etc.
    boxConstraint = logspace(-3, 3, 7);
    bestCVScore = Inf;
    bestBoxConstraint = boxConstraint(1);
    
    for C = boxConstraint
        % Train the SVM model with the current hyperparameter
        svmModel = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear', 'BoxConstraint', C);
        
        % Perform k-fold cross-validation
        cvModel = crossval(svmModel, 'KFold', k);
        
        % Calculate the cross-validation classification accuracy
        cvAccuracy = 1 - kfoldLoss(cvModel, 'LossFun', 'ClassifError');
        
        % Check if we have found a better hyperparameter
        if cvAccuracy < bestCVScore
            bestCVScore = cvAccuracy;
            bestBoxConstraint = C;
        end
    end
    
    % Refit the SVM model with the best hyperparameter on the full training set
    finalSvmModel = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear', 'BoxConstraint', bestBoxConstraint);
    
    % Perform prediction on the test set
    YPred = predict(finalSvmModel, XTest);
    
    % Calculate the test set classification accuracy
    testAccuracy = sum(YPred == YTest) / length(YTest);
    
    % Assign the best hyperparameters to the output struct
    hyperParam = struct('BoxConstraint', bestBoxConstraint);
    
    % Assign the best cross-validation accuracy to the output
    crossValAccuracy = bestCVScore;
end
