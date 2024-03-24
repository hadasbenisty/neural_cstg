% This script aims to find, given a diffusion map of N elements, the best D
% elements that show the most of the learning process. 

%% Loading Diffusion Map data
load("..\data\paths\paths.mat")
load(fullfile(inputs_path, "diffusion_map_analysis.mat"));

%% PreProcessing
smooth_func = @(vec) sgolayfilt(vec, ...
        2, 51);
smooth_sf = smooth_func(sflabels);
%% Smoothing the data
diffusion_map_smoothed = NaN(size(diffusion_map));
for dim = 1:size(diffusion_map, 2)
    diffusion_map_smoothed(:, dim) = smooth_func(diffusion_map(:, dim));
    diffusion_map_smoothed(:,dim) = (diffusion_map_smoothed(:, dim) - ...
        min(diffusion_map_smoothed(:, dim))) ./ ...
                                (max(diffusion_map_smoothed(:, dim)) - ...
                                min(diffusion_map_smoothed(:, dim))); 
end

% Plot an example
figure;
plot(diffusion_map_smoothed(:, 3));

%% Colculating Correlations Between Diffusion Map and SF
diff_map_sf_corr = NaN(1, size(diffusion_map, 2));
for dim = 1:size(diffusion_map, 2)
    diff_map_sf_corr(dim) = abs(corr(diffusion_map_smoothed(:, dim), ...
        smooth_sf));
end

corr_diff_sf_fig = figure;
bar(diff_map_sf_corr);
title('Correlation Between Diffusion Map Element and Sucess Failure');
xlabel('Dimension (element) #');
ylabel('Correlation (absolute value)');

[~, dim_importance_inds] = sort(diff_map_sf_corr, 'descend');

%% Trying to find the most important features
% X = diffusion_map_smoothed; Y = sflabels;
% % Assuming X is your features matrix and Y is your labels vector
% tree = fitctree(X, Y);
% 
% % Get the importance of each feature
% importance = tree.predictorImportance;
% 
% % Sort the importance in descending order
% [sortedImportance,sortedIndices] = sort(importance,'descend');
% 
% % Now sortedIndices contains the indices of the features sorted by their importance
%% Average Diffusion Map VS. Avg Success rate
last_train_stage = max(train_stage); first_train_stage = min(train_stage);
diffusion_map_averaged = NaN(last_train_stage - first_train_stage + 1, ...
    size(diffusion_map, 2));
for session = first_train_stage:last_train_stage
    diffusion_map_averaged(session, :) = mean(diffusion_map_smoothed(...
        train_stage == session, :));
    diffusion_map_averaged(session, :) = ...
        ((diffusion_map_averaged(session, :) - ...
        min(diffusion_map_averaged(session, :))) ./ ...
        (max(diffusion_map_averaged(session, :) - ...
        min(diffusion_map_averaged(session, :)))));
end

% Plot an example
figure;
plot(avg_suc_rate, 'DisplayName', 'Average Success Rate');
hold on;
plot(diffusion_map_averaged(:, 3), 'DisplayName', 'Average Diffusion Map');
hold off;
legend;

%% Calculate Variance Along Trials
diff_map_var = var(diffusion_map_smoothed, 0, 1);

var_diff_map_fig = figure;
plot(diff_map_var);
title('The variance for each trail over dimensions');
xlabel('Dim #');
ylabel('Variance');