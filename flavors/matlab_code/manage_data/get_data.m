function [X, Y] = get_data(imagingData_samples, BehaveData, f1, f2)
if isfield(BehaveData, f1) && isfield(BehaveData, f2)
    Y1 = BehaveData.(f1).indicatorPerTrial;
    Y2 = BehaveData.(f2).indicatorPerTrial;
    Y = Y1-Y2;
    
    X = imagingData_samples(:, :, Y ~= 0);
    Y = Y(Y ~= 0);
else
    X = [];
    Y = [];
end
end
