function estimatedLevel = KNNEstimatedLevel(X, Y, k, foldsNum)
%KNNEstimatedLevel Trains a KNN tree model and tests it using cross
%validation
% ---- Inputs ----
% X - a set of samples, the size of features x samples
% Y - a set of labels, the size of 1 x samples. The train sessions.
% k - an integer, the number of neighbors we accept for the model.
% foldsNum - an integer, the number of folds we split the data when
% calculating the accuracy.
% ---- Outputs ----
% estimatedLevel - the estimated level of the mice at each train session.

    expertIndex = max(Y); naiveIndex = min(Y);
    % Find the train data
    Xtr = X(:, Y == naiveIndex); Xtr = [Xtr, X(:, Y == expertIndex)];
    % Xte = X(:, Y ~= naiveIndex && Y ~= expertIndex);
    Ytr = zeros(sum(Y == expertIndex), 1); Ytr = [Ytr; ...
        zeros(sum(Y == naiveIndex), 1)];
    Ytr(end-sum(Y == expertIndex):end) = 1;

    % Train the model
    knnModel = fitcknn(Xtr', Ytr, 'NumNeighbors', k);

    % Predicting on the rest of the data
    predictionsKNN = predict(knnModel, X');

    % Calculate average expert level for each train session
    estimatedLevel = zeros(size(unique(Y)));
    for trainSession = naiveIndex:expertIndex
        estimatedLevel(trainSession) = mean(...
            predictionsKNN(Y == trainSession));
    end
end