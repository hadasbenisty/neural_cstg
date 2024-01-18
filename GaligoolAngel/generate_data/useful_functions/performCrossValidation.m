function avgAccuracy = performCrossValidation(X, Y, foldsNum, model)
    % Initialize variables
    cv = cvpartition(Y, 'KFold', foldsNum);
    accuracies = zeros(cv.NumTestSets, 1);

    % Loop over folds
    for i = 1:cv.NumTestSets
        % Split the data into training and testing
        trainIdx = cv.training(i);
        testIdx = cv.test(i);

        X_train = X(trainIdx, :);
        Y_train = Y(trainIdx, :);

        X_test = X(testIdx, :);
        Y_test = Y(testIdx, :);

        % Train the model
        trainedModel = model(X_train, Y_train);

        % Predict the labels of the test data
        Y_pred = predict(trainedModel, X_test);

        % Calculate the accuracy
        accuracies(i) = sum(Y_pred == Y_test) / length(Y_test);
    end

    % Calculate the average accuracy
    avgAccuracy = mean(accuracies);
end
