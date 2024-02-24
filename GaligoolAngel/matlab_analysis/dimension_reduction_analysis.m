% Data Organization
%% Loading Data
labelsCTRL = {'train_1' 'train_2' 'train_3' 'train_4' ...
        'train_5' 'train_6' 'train_7'};
labelsCNO = {'train_1' 'CNO_2' 'CNO_3' 'CNO_4' ...
        'train_5' 'train_6' 'train_7'};

animalsnames = {'DT141' 'DT155'};
animalsLabels = [0 1];

chosen_animal = 1; % or 1
disp("loading data")
datapath = '../data/';
load(fullfile(datapath, animalsnames{chosen_animal}, 'data.mat'));
load('..\data\paths\paths.mat');
%% PreProcessing
last_train_session = 7;
data_all = data_all(:,:, train_stage <= last_train_session);
train_stage = train_stage(train_stage <= 7);
sflabels = sflabels(train_stage <= 7);
training_lut = training_lut(1:last_train_session);
CC = calcCorrelationMatrix(permute(data_all(:, 20:end, :), [2,1,3])); % ...
% Calculating the correlation matrix in the last 4 seconds of the
% measurements.
avg_suc_rate = zeros(size(training_lut))';
for session = 1:length(training_lut)
    indicator = train_stage == session;
    avg_suc_rate(session) = sum(sflabels(train_stage == session)) / ...
        length(sflabels(train_stage == session));
    avg_suc_rate(session) = max(avg_suc_rate(session), 1 - ...
        avg_suc_rate(session));
end

%% CC Analysis
% SVM All
CC_features = getLowerHalf(CC);
CC_features_ext = permute(CC_features, [1, 3, 2]);


[~, estimated_level_CC] = naive_expert_svm_CC ...
        (CC_features_ext, train_stage, training_labels_lut); 

fig_svm_cc = figure;
plot(estimated_level_CC, 'DisplayName', 'Estimated Level');
title(['The Estimated Expertee Along The Train Sessions for animal no.' ...
     , num2str(chosen_animal)])
xlabel('Train Session #')
legend;

% PCA Dim Analysis
pca_cc_features = CC_features';
[coeff, score, latent, tsquared, explained, mu] = ...
    pca(pca_cc_features);

indices = 2:1:50;
estimated_level_matrix_euclid = zeros([size(estimated_level_CC, 1), ...
    length(indices)]);
for ii = indices
    lower_dim_vecs = coeff(:, 1:ii)';
    svm_lower_dim = pca_cc_features * lower_dim_vecs';
    svm_lower_dim = permute(svm_lower_dim, [2, 3, 1]);

    % Changing data_all to CC
    [~, estimated_level_matrix_euclid(:, ii)] = naive_expert_svm_CC ...
        (svm_lower_dim, train_stage, training_labels_lut); 

end

fig_pca = figure;
imagesc(estimated_level_matrix_euclid);
title(['The Estimated Expertee Along The Train Sessions for animal no.' ...
    , num2str(chosen_animal)])
ylabel('Train Session')
xlabel('The Number Of Dimensions Left')
colormap('jet')
colorbar;

% PCA 3D Reduction
coeffs_3d = coeff(:, 1:3)';
PCA_lower_dims = pca_cc_features * coeffs_3d';

pca_3d_figure = figure;
cmap = hsv(max(train_stage)); colors = cmap(train_stage, :);
scatter3(PCA_lower_dims(:, 1), PCA_lower_dims(:, 2), PCA_lower_dims(:,3) ...
    ,16 ,colors, 'filled');
xlabel('First PCA Component')
ylabel('Second PCA Component')
zlabel('Third PCA Component')
title('First 3 Components of the PCA On Our Correlation Data')

% Dimension Reduction Map Diffusion
indices = 2:1:50;
estimated_level_matrix = zeros([size(estimated_level_CC, 1), length(indices)]);
chanceLevel_vector = zeros(length(indices));
dR1 = calc_Rdist(CC);
[aff_mat, ~] = CalcInitAff2D( dR1, 5);
configParams.maxInd = 5;
for ii = indices
    diffusion_map = calcDiffusionMap(aff_mat,configParams, ii); % Remove last param for default
    
    diffusion_map = permute(diffusion_map, [1, 3, 2]);
    
    % Changing data_all to CC
    [~, estimated_level_matrix(:, ii)] = naive_expert_svm_CC ...
        (diffusion_map, train_stage, training_labels_lut); 
end

fig_diffusion_map = figure;
imagesc(estimated_level_matrix);
title(['The Estimated Expertee Along The Train Sessions for animal no.' ...
    , num2str(chosen_animal)])
ylabel('Train Session')
xlabel('The Number Of Dimensions Left')
colormap('jet');
colorbar;

% Diffusion Map 3D reduction
configParams.maxInd = 20;
diffusion_map = (calcDiffusionMap(aff_mat,configParams, 20))';
diffusion_map_3d_fig = figure;
cmap = hsv(max(train_stage)); colors = cmap(train_stage, :);
scatter3(diffusion_map(:, 1), diffusion_map(:, 2),diffusion_map(:,3) ...
    ,16 ,colors, 'filled');
xlabel('First Diffusion Map Component')
ylabel('Second Diffusion Map Component')
zlabel('Third Diffusion Map Component')
title('First 3 Components of the Diffusion Map On Our Correlation Data')


%% SF Analysis
if size(avg_suc_rate, 1) ~= 1
    avg_suc_rate = avg_suc_rate';
end
% SF Each
sf_lut = {'fail','suc'};
model_accuracy = zeros(size(training_lut));
CC_features_squeezed = squeeze(CC_features)';
for train_stage_num = min(train_stage):max(train_stage)
    model_accuracy(train_stage_num) = svm( ...
        CC_features_squeezed(train_stage==train_stage_num, :), ...
        sflabels(train_stage==train_stage_num), 3, 1);
end

fig_svm_sf_each = figure;
plot(model_accuracy - avg_suc_rate, 'DisplayName', "Model's Accuracy -" + ...
    " average sucess rate");
hold on;
% plot(avg_suc_rate, 'DisplayName', "Average Sucess Rate");
legend;
title('The SVM prediction sucess rate trained on each session')
xlabel('Train Stage')
ylabel('Precentage Rate');
hold off;

% SF All
[model_accuracy, test_accuracy, SVMMOdel, test_data, validation_data] = ...
    svm(squeeze(CC_features)', sflabels, 5, 0.2);

predictions_sf = predict(SVMMOdel, test_data(:, 1:end-2));

for train = min(train_stage):max(train_stage)
    % rate = dot(predictions_sf(train_stage(test_data(:,end)) == train), ...
    %     sflabels(train_stage(test_data(:,end)) == train)) / ...
    %     sum(sflabels(train_stage(test_data(:,end)) == train));
    [TP, TN, FP, FN] = calculateConfusionMatrixElements(...
        predictions_sf(train_stage(test_data(:,end)) == train), ...
        sflabels(train_stage(test_data(:,end)) == train));
    rate = calculateAccuracy(TP, TN, FP, FN);
    sucess_rate_sf(train) = max(rate, 1- rate);
end

fig_svm_sf_all = figure;
plot(sucess_rate_sf - avg_suc_rate, 'DisplayName', "Model's Accuracy" + ...
    " - avg sucess rate");
hold on;
% plot(avg_suc_rate, 'DisplayName', "Average Sucess Rate");
legend;
title(['Sucess rate of svm trained on all of the sessions', ...
    'animal no.', num2str(chosen_animal)]);
xlabel('Train Session')
ylabel('Sucess Rate')
hold off;

% SF Last
CC_features = squeeze(CC_features);
train_stage_num = max(train_stage);
[model_accuracy_last, test_accuracy_last, SVMMOdel_last, ...
    test_data_last, validation_data] = ...
    svm(CC_features(train_stage(1:end-1)==train_stage_num, :), ...
        sflabels(train_stage(1:end-1)==train_stage_num), 5, 0.4);

sucess_rate_sf_last = [];
for train = min(train_stage):max(train_stage)
    predictions_sf_last = predict(SVMMOdel_last, CC_features(...
        train_stage(1:end) == train, :));
    [TP, TN, FP, FN] = calculateConfusionMatrixElements(...
        predictions_sf_last, ...
        sflabels(train_stage == train));
    rate = calculateAccuracy(TP, TN, FP, FN);
    sucess_rate_sf_last = [sucess_rate_sf_last, max(rate, 1 - rate)];
    % sucess_rate_sf(train) = performCrossValidation( ...
    %     CC_features(:, train_stage == train), ...
    %     sflabels(train_stage == train), 5, SVMMOdel_last);

end

% Plotting the result
fig_svm_sf_last = figure;
plot(sucess_rate_sf_last - avg_suc_rate, 'DisplayName', ...
    "Model's Accuracy - Average Sucess Rate");
hold on;
% plot(avg_suc_rate, "DisplayName", "Average Sucess Rate");
legend;
title(['The sucess rate of an SVM model trained on the last session' ...
    ' on the ' ...
    'other sessions animal no.', num2str(chosen_animal)]);
xlabel('Train Session')
ylabel('Sucess Rate')
hold off;

%% TSNE Analysis
tsne_dist_func = @(x, y) SpsdDist(vectorToSymMatrix(x), ...
    vectorToSymMatrix(y), 5);
normalized_cc = zscore(CC_features');
Y = tsne(CC_features', 'NumDimensions', 3, 'Distance', ...
    tsne_dist_func, ...
    'Exaggeration', length(unique(train_stage)));

tsne_algo_fig = figure;
cmap = hsv(max(train_stage)); colors = cmap(train_stage, :);
scatter3(Y(:,1), Y(:,2), Y(:,3), 16, colors, "filled");
title('TSNE on our data (correlations)')


%% Best Time Window Analysis
% partitions = 5;
% time_window = size(data_all, 2) ./ partitions; % only 20 percent of the time
% times = size(data_all,2) * linspace(1, partitions, partitions) ./ ...
%     partitions;
% estimated_level_CC_time_windows = [];
% for ii = 1:partitions
%     data_all_eff = data_all(:, times(ii):end, :);
%     CC = calcCorrelationMatrix(permute(data_all_eff, [2,1,3]));
%     CC_features_ext = permute( getLowerHalf(CC), [1, 3, 2]);
%     [~, estimated_level_CC_time_windows(ii)] = naive_expert_svm_CC ...
%         (CC_features_ext, train_stage, training_labels_lut); 
% end

%% Saving Figures
animal_num_str = num2str(chosen_animal);
if ~isfolder(fullfile(results_path, animal_num_str))
    mkdir(fullfile(results_path, animal_num_str));
end
saveas(fig_svm_cc, fullfile(results_path, animal_num_str, ...
    'fig_svm_cc'), 'jpg');
saveas(fig_pca, fullfile(results_path,animal_num_str, 'fig_pca'), 'jpg');
saveas(fig_diffusion_map, fullfile(results_path, animal_num_str, ...
    'fig_diffusion_map'), ...
    'jpg');
saveas(fig_svm_sf_all, fullfile(results_path, animal_num_str, ...
    'fig_svm_sf_all'), 'jpg');
saveas(fig_svm_sf_each, fullfile(results_path, animal_num_str, ...
    'fig_svm_sf_each'), 'jpg');
saveas(fig_svm_sf_last, fullfile(results_path, animal_num_str, ...
    'fig_svm_sf_last'), 'jpg');
saveas(pca_3d_figure, fullfile(results_path, animal_num_str, ...
    'fig_PCA_3d_CC'), 'jpg');
saveas(diffusion_map_3d_fig,  fullfile(results_path, animal_num_str, ...
    'fig_diffusion_map_3d_CC'), 'jpg');
saveas(tsne_algo_fig,  fullfile(results_path, animal_num_str, ...
    'tsne_3d_CC'), 'jpg');