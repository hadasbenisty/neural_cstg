clear
% 
% cd 'libsvm/matlab';
% make;
datapath = '../data';
% addpath(genpath("libsvm"))
labelsCTRL = {'train_1' 'train_2' 'train_3' 'train_4' 'train_5' 'train_6' 'train_7'};
labelsCNO = {'train_1' 'CNO_2' 'CNO_3' 'CNO_4' 'train_5' 'train_6' 'train_7'};
addpath(genpath('.\'));

animalsnames = {'DT141' 'DT155'};
animalsLabels = [0 1];

chosen_animal = 1; % or 1
disp("loading data")
load(fullfile(datapath, animalsnames{chosen_animal}, 'data.mat'));

%% svm - beginner vs. expert
disp("svm beginner vs expert")
[chanceLevel, estimated_level] = naive_expert_svm(data_all, train_stage, training_labels_lut); 
figure;imagesc(linspace(-3.5, 7.5, 23), 1:7, estimated_level, [0 1]); set(gca, 'YtickLabels', labelsCTRL);
xlabel('Time [sec]'); ylabel('Session');title('Prediction probability as Expert')
colormap jet;colorbar;
saveas(gcf, '../results/beginner_vs_expert.jpg', 'jpg');

%% pca trajectories by outcome
disp("pca trajectories by outcome")
trajectories_analysis_sf(data_all, sflabels, train_stage, training_labels_lut);

%% centroids
disp("Riemannian centroids")
[mC, CC] = calc_centroids(data_all, train_stage);
dR1 = calc_Rdist(CC);
[aff_mat, sigma] = CalcInitAff2D( dR1, 5);
configParams.maxInd = 5;
configParams.plotResults = 1;
diffusion_map_main = calcDiffusionMap(aff_mat,configParams); 
% Remove last param for default
figure;scatter3(diffusion_map_main(1, :), diffusion_map_main(2, :), ...
    diffusion_map_main(3, :), 10, train_stage, 'filled');
colormap jet;
title(['Correlation matrices embedded by Diffusion Map,' ...
    ' Colors - Training Session'])
saveas(gcf, '../results/corr_matrices.jpg', 'jpg');

dR = calc_Rdist(mC);
disp('diffusion embedding')
aff_mat_centroids = CalcInitAff2D( dR, 5);

diffusion_map_centroids = calcDiffusionMap(aff_mat_centroids,configParams);
figure;scatter3(diffusion_map_centroids(1, :), diffusion_map_centroids(2, :), diffusion_map_centroids(3, :), 10, 1:7, 'filled');
hold all; plot3(diffusion_map_centroids(1, :), diffusion_map_centroids(2, :), diffusion_map_centroids(3, :), 'k')
title('Centroids matrices embedded by Diffusion Map, Colors - Training Session');
saveas(gcf, '../results/centroids.jpg', 'jpg');
%% Degree
centality_vals = get_centrality(mC);
[~, ic] = sort(centality_vals.degree(:, 7));
figure;imagesc(centality_vals.degree(ic, :));
ylabel('Cells'); xlabel('Training Session');
title('Degree')
saveas(gcf, '../results/degree.jpg', 'jpg');


