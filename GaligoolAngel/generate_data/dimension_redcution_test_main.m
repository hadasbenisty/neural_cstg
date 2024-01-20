%% Load data from before
addpath(genpath('.\')); % Adding all the functions from the folders
create_data_bool = 1; % Whether we want to build the data or
% use the one we have
if create_data_bool % Loading the raw data
    labelsCTRL = {'train_1' 'train_2' 'train_3' 'train_4' ...
        'train_5' 'train_6' 'train_7'};
    labelsCNO = {'train_1' 'CNO_2' 'CNO_3' 'CNO_4' ...
        'train_5' 'train_6' 'train_7'};
    
    animalsnames = {'DT141' 'DT155'};
    animalsLabels = [0 1];
    
    chosen_animal = 1; % or 1
    disp("loading data")
    datapath = 'data/';
    load(fullfile(datapath, animalsnames{chosen_animal}, 'data.mat'));
else % Loading the processed data
    load('workspace.mat');
end
%% Cutting the data in the 7th train session
last_train_session = 7;
data_all = data_all(:,:, train_stage <= last_train_session);
train_stage = train_stage(train_stage <= 7);
sflabels = sflabels(train_stage <= 7);
training_lut = training_lut(1:last_train_session);

%% Calculating the Correlation Matrix
CC = calcCorrelationMatrix(permute(data_all(:, 240:end, :), [2,1,3]));

%% Ttying SVM CC
% CC_features = zeros(size(CC,2)^2, 1);
CC_features = getLowerHalf(CC);
CC_features = permute(CC_features, [1, 3, 2]);


[~, estimated_level_CC] = naive_expert_svm_CC ...
        (CC_features, train_stage, training_labels_lut); 

figure;
imagesc(estimated_level_CC);
title(['The Estimated Expertee Along The Train Sessions for animal no.' ...
     , num2str(chosen_animal)])
ylabel('Train Session')
colormap('jet');
colorbar;

%% Trying PCA reduction

pca_cc_features = squeeze(CC_features)';
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

figure;
imagesc(estimated_level_matrix_euclid);
title(['The Estimated Expertee Along The Train Sessions for animal no.' ...
    , num2str(chosen_animal)])
ylabel('Train Session')
xlabel('The Number Of Dimensions Left')
colormap('jet')
colorbar;

%% Choosing the right number of dimensions reduction
indices = 2:1:50;
estimated_level_matrix = zeros([size(estimated_level_CC, 1), length(indices)]);
chanceLevel_vector = zeros(length(indices));
dR1 = calc_Rdist(CC);
[ aff_mat,sigma ] = CalcInitAff2D( dR1, 5 );
configParams.maxInd = 5;
for ii = indices
    diffusion_map = calcDiffusionMap(aff_mat,configParams, ii); % Remove last param for default
    
    diffusion_map = permute(diffusion_map, [1, 3, 2]);
    
    % Changing data_all to CC
    [~, estimated_level_matrix(:, ii)] = naive_expert_svm_CC ...
        (diffusion_map, train_stage, training_labels_lut); 
end

figure;
imagesc(estimated_level_matrix);
title(['The Estimated Expertee Along The Train Sessions for animal no.' ...
    , num2str(chosen_animal)])
ylabel('Train Session')
xlabel('The Number Of Dimensions Left')
colormap('jet');
colorbar;

%% Calculating Average Sucess Rate
avg_suc_rate = zeros(size(training_lut))';
for session = 1:length(training_lut)
    indicator = train_stage == session;
    avg_suc_rate(session) = sum(sflabels(train_stage == session)) / ...
        length(sflabels(train_stage == session));
end


%% SF SVM
sf_lut = {'fail','suc'};
estimated_level_sf = zeros(length(sf_lut), length(training_lut));
model_accuracy = zeros(size(training_lut));
CC_features_squeezed = squeeze(CC_features)';
for train_stage_num = min(train_stage):max(train_stage)
    % [chance_level_sf, estimated_level_sf(:, train_stage_num)] = ...
    %     naive_expert_svm_CC(...
    %     CC_features(:, :, train_stage==train_stage_num), ...
    %     sflabels(train_stage==train_stage_num)+1, sf_lut);
    model_accuracy(train_stage_num) = svm( ...
        CC_features_squeezed(train_stage==train_stage_num, :), ...
        sflabels(train_stage==train_stage_num), 5, 1);
    % model_BT_accuracy(train_stage_num) = baggedTreesModel(...
    %     CC_features_squeezed(train_stage==train_stage_num, :), ...
    %     sflabels(train_stage==train_stage_num), 0.1, 100);
end

figure;
plot(model_accuracy, 'DisplayName', "Model's Accuracy");
hold on;
plot(avg_suc_rate, 'DisplayName', "Average Sucess Rate");
legend;
title('The SVM prediction sucess rate over training sessions')
xlabel('Train Stage')
ylabel('Precentage Rate');
hold off;

%% 
[model_accuracy, test_accuracy, SVMMOdel, test_data, validation_data] = ...
    svm(CC_features', sflabels, 5, 0.2);

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

figure;
plot(sucess_rate_sf, 'DisplayName', "Model's Accuracy");
hold on;
plot(avg_suc_rate, 'DisplayName', "Average Sucess Rate");
legend;
title('Sucess rate of svm over train sessions (cross validation score)');
xlabel('Train Session')
ylabel('Sucess Rate')
hold off;

%% Training a model on the last train session and testing on all the others
CC_features = squeeze(CC_features);
train_stage_num = max(train_stage);
[model_accuracy_last, test_accuracy_last, SVMMOdel_last, ...
    test_data_last, validation_data] = ...
    svm(CC_features(train_stage(1:end-1)==train_stage_num, :), ...
        sflabels(train_stage(1:end-1)==train_stage_num), 5, 1);

for train = min(train_stage):max(train_stage)
    predictions_sf_last = predict(SVMMOdel_last, CC_features(...
        train_stage(1:end-1) == train, :));
    [TP, TN, FP, FN] = calculateConfusionMatrixElements(...
        predictions_sf_last(train_stage(test_data_last(:,end)) == train), ...
        sflabels(train_stage(test_data_last(:,end)) == train));
    rate = calculateAccuracy(TP, TN, FP, FN);
    sucess_rate_sf_last(train) = max(rate, 1 - rate);
    % sucess_rate_sf(train) = performCrossValidation( ...
    %     CC_features(:, train_stage == train), ...
    %     sflabels(train_stage == train), 5, SVMMOdel_last);

end

% Plotting the result
figure;
plot(sucess_rate_sf_last, 'DisplayName', "Model's Accuracy");
hold on;
plot(avg_suc_rate, "DisplayName", "Average Sucess Rate");
legend;
title(['The sucess rate of an SVM model trained on the last session' ...
    ' on the ' ...
    'other sessions'])
xlabel('Train Session')
ylabel('Sucess Rate')
hold off;

%% Calculating Y 
output_dim = 2;
dR1 = calc_Rdist(CC);
[ aff_mat,sigma ] = CalcInitAff2D( dR1, 5 );
configParams.maxInd = 5;
diffusion_map = calcDiffusionMap(aff_mat,...
    configParams, output_dim + 1); % Remove last param for default

%% Saving Data
load('data\paths\paths.mat');
y = squeeze(diffusion_map);
y = y ./ mean(abs(y), 2);
features = squeeze(CC_features);
features = features ./ mean(features, 2);
features_pca = (pca_cc_features * coeff(:, 1:30))';
features_pca = features_pca ./ mean(features_pca, 2);

save(fullfile(inputs_path, 'dataset'), 'y', 'features_pca', 'features');
