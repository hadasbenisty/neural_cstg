function accuracy = calculateAccuracy(TP, TN, FP, FN)
    % This function calculates the accuracy of a binary classification model.
    % Inputs:
    %   TP - True Positives
    %   TN - True Negatives
    %   FP - False Positives
    %   FN - False Negatives
    %
    % Output:
    %   accuracy - The accuracy of the model

    % Calculate the accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN);
end
