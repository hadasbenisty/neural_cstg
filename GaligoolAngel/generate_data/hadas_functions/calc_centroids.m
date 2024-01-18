function [mC, CCall] = calc_centroids(data_all, train_stage)
% ---- Description ----
% ---- Inputs ----
% data_all - a tensor the size of neurons x time_samples x trials. Holding
% the activity of the neurons for each time sample at each trial.
% train_stage - a vector the size of trials x  1. Holding for each trial to
% which session it belongs.
% ---- Outputs ----
% mC - 
% CCall - 
% ---- Inmportant Local Parameters ----
% clusters - a vector the size of number_of_different_sessions x 1

    CCall = [];
    clusters = unique(train_stage);
    for ci = 1:length(clusters)
        x = data_all(:, 20:end, train_stage == ci);
        CC=[];r=[];e=[];
        for kk=1:size(x, 3)
            CC(:, :, kk) = corr(x(:, :, kk)');
            e(:,kk) = sort(eig(CC(:,:,kk)), 'descend');
            r(kk) = find(e(:,kk) < median(e(:,kk) ), 1);
        end
        CCall = cat(3, CCall, CC);
        r = min(r);
        mC(:, :, ci) = SpsdMean(CC, r);
    end
end




