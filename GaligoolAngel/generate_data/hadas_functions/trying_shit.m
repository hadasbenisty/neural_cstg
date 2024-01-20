% New SVM weird
% Changing data_all to CC
[chanceLevel_cc, estimated_level_cc] = naive_expert_svm_CC(CC, ...
 train_stage, training_labels_lut); 

%% Changing data_all to diffusion_map
diffusion_map_fake = zeros([size(diffusion_map), 2]);
diffusion_map_fake(:, :, 1) = diffusion_map;
diffusion_map_fake(:, :, 2) = diffusion_map;
diffusion_map_fake = permute(diffusion_map_fake, [1, 3, 2]);

% Changing data_all to CC
[chanceLevel_cc, estimated_level_cc] = naive_expert_svm_CC ...
    (diffusion_map_fake, train_stage, training_labels_lut); 
