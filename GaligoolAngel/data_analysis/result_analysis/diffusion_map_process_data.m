% Basic Script to Process Processed DAta From Python
data_path = ['C:\Users\hadas-stud-group2\OneDrive' ...
    ' - Technion\First Degree\Project A\results\processed_data.mat'];
load(data_path);

%% Diffusion Map On All Features
dR1 = calc_Rdist(CC);
[aff_mat, ~] = CalcInitAff2D( dR1, 5);
configParams.maxInd = 5;
diffusion_map = (calcDiffusionMap(aff_mat,configParams, 4))';
diffusion_map_3d_fig = figure;
cmap = hsv(max(train_stage)); colors = cmap(train_stage, :);
scatter3(diffusion_map(:, 1), diffusion_map(:, 2),diffusion_map(:,3) ...
    ,16 ,colors, 'filled');
xlabel('First Diffusion Map Component')
ylabel('Second Diffusion Map Component')
zlabel('Third Diffusion Map Component')
title('First 3 Components of the Diffusion Map On Our Correlation Data')

%% Difusion map dynamic correalations
data_imp = data_all(neurons_indices + 1, :, :);
CC_imp = calcCorrelationMatrix(permute(data_imp(:, 20:end, :), [2,1,3]));

dR1_imp = calc_Rdist(CC_imp);
[aff_mat, ~] = CalcInitAff2D( dR1_imp, 5);
configParams.maxInd = 5;
diffusion_map_imp = (calcDiffusionMap(aff_mat,configParams, 8))';
diffusion_map_3d_fig = figure;
cmap = hsv(max(train_stage)); colors = cmap(train_stage, :);
scatter3(diffusion_map_imp(:, 1), diffusion_map_imp(:, 2),diffusion_map_imp(:,3) ...
    ,16 ,colors, 'filled');
xlabel('First Diffusion Map Component')
ylabel('Second Diffusion Map Component')
zlabel('Third Diffusion Map Component')
title('First 3 Components of the Diffusion Map On Our Correlation Data')

% SVM Test
CC_features_imp = getLowerHalf(CC_imp);
CC_features_ext_imp = permute(CC_features_imp, [1 3 2]);
[~, estimated_level_CC_imp] = naive_expert_svm_CC ...
        (CC_features_ext_imp, train_stage, training_labels_lut);
diff_map_feat_ext_imp = permute(diffusion_map_imp', [1 3 2]);
[~, estimated_level_CC_diff_map_imp] = naive_expert_svm_CC ...
        (diff_map_feat_ext_imp, train_stage, training_labels_lut);
figure;
plot(estimated_level_CC_imp); hold on;
plot(estimated_level_CC_diff_map_imp);

%% Non Important 
% Given data_all and neurons_indices
% Convert neurons_indices to 1-based indexing if it's currently 0-based
% If it's already 1-based, you can skip this step
neurons_indices = neurons_indices + 1;

% Get the total number of neurons
total_neurons = size(data_all, 1);

% Create an array of all neuron indices
all_indices = 1:total_neurons;

% Find the indices of neurons that are not in neurons_indices
not_imp_indices = setdiff(all_indices, neurons_indices);

% Select the data for non-important neurons
data_not_imp = data_all(not_imp_indices, :, :);
CC_not_imp = calcCorrelationMatrix(permute(data_not_imp(:, 20:end, :), [2,1,3]));

dR1_not_imp = calc_Rdist(CC_not_imp);
[aff_mat, ~] = CalcInitAff2D( dR1_not_imp, 5);
configParams.maxInd = 5;
diffusion_map_not_imp = (calcDiffusionMap(aff_mat,configParams, 8))';
diffusion_map_3d_fig = figure;
cmap = hsv(max(train_stage)); colors = cmap(train_stage, :);
scatter3(diffusion_map_not_imp(:, 1), diffusion_map_not_imp(:, 2),diffusion_map_not_imp(:,3) ...
    ,16 ,colors, 'filled');
xlabel('First Diffusion Map Component')
ylabel('Second Diffusion Map Component')
zlabel('Third Diffusion Map Component')
title('First 3 Components of the Diffusion Map On Our Correlation Data')

% SVM Test
CC_features_not_imp = getLowerHalf(CC_not_imp);
CC_features_ext_not_imp = permute(CC_features_not_imp, [1 3 2]);
[~, estimated_level_CC_not_imp] = naive_expert_svm_CC ...
        (CC_features_ext_not_imp, train_stage, training_labels_lut);
diff_map_feat_ext_not_imp = permute(diffusion_map_not_imp', [1 3 2]);
[~, estimated_level_CC_diff_map_not_imp] = naive_expert_svm_CC ...
        (diff_map_feat_ext_not_imp, train_stage, training_labels_lut);
figure;
plot(estimated_level_CC_not_imp); hold on;
plot(estimated_level_CC_diff_map_not_imp);


%% Realy not important at all
% Given data_all and neurons_indices
% Convert neurons_indices to 1-based indexing if it's currently 0-based
% If it's already 1-based, you can skip this step
neurons_indices_mean = neurons_indices_mean + 1;

% Get the total number of neurons
total_neurons = size(data_all, 1);

% Create an array of all neuron indices
all_indices = 1:total_neurons;

% Find the indices of neurons that are not in neurons_indices
not_imp_indices_at_all = setdiff(all_indices, neurons_indices_mean);

% Select the data for non-important neurons
data_not_imp_at_all = data_all(not_imp_indices_at_all, :, :);
CC_not_imp_at_all = calcCorrelationMatrix(permute(data_not_imp_at_all(:, 20:end, :), [2,1,3]));

dR1_not_imp_at_all = calc_Rdist(CC_not_imp_at_all);
[aff_mat, ~] = CalcInitAff2D( dR1_not_imp_at_all, 5);
configParams.maxInd = 5;
diffusion_map_not_imp_at_all = (calcDiffusionMap(aff_mat,configParams, 8))';
diffusion_map_3d_fig = figure;
cmap = hsv(max(train_stage)); colors = cmap(train_stage, :);
scatter3(diffusion_map_not_imp_at_all(:, 1), diffusion_map_not_imp_at_all(:, 2),diffusion_map_not_imp_at_all(:,3) ...
    ,16 ,colors, 'filled');
xlabel('First Diffusion Map Component')
ylabel('Second Diffusion Map Component')
zlabel('Third Diffusion Map Component')
title('First 3 Components of the Diffusion Map On Our Correlation Data')

% SVM Test
CC_features_not_imp_at_all = getLowerHalf(CC_not_imp_at_all);
CC_features_ext_not_imp_at_all = permute(CC_features_not_imp_at_all, [1 3 2]);
[~, estimated_level_CC_not_imp_at_all] = naive_expert_svm_CC ...
        (CC_features_ext_not_imp_at_all, train_stage, training_labels_lut);
diff_map_feat_ext_not_imp_at_all = permute(diffusion_map_not_imp_at_all', [1 3 2]);
[~, estimated_level_CC_diff_map_not_imp_at_all] = naive_expert_svm_CC ...
        (diff_map_feat_ext_not_imp_at_all, train_stage, training_labels_lut);
figure;
plot(estimated_level_CC_not_imp_at_all); hold on;
plot(estimated_level_CC_diff_map_not_imp_at_all);

%% Diffusion Map Big Mean
% CC_imp = zeros(size(important_mus, 1), size(CC_features, 2));
for ii = 1:size(CC_features, 2)
    CC_mean(:, :, ii) = vectorToSymMatrix(CC_features(mus_mean, ii));
end

dR1_mean = calc_Rdist(CC_imp);
[aff_mat, ~] = CalcInitAff2D( dR1_mean, 5);
configParams.maxInd = 5;
diffusion_map = (calcDiffusionMap(aff_mat,configParams, 8))';
diffusion_map_3d_fig = figure;
cmap = hsv(max(train_stage)); colors = cmap(train_stage, :);
scatter3(diffusion_map(:, 1), diffusion_map(:, 2),diffusion_map(:,3) ...
    ,16 ,colors, 'filled');
xlabel('First Diffusion Map Component')
ylabel('Second Diffusion Map Component')
zlabel('Third Diffusion Map Component')
title('First 3 Components of the Diffusion Map On Our Correlation Data')

%% Saving Results
