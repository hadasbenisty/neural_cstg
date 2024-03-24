function [output] = save_data_for_cstg(CC_features,train_stage,diff_map,smooth,datafile_name)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
features=CC_features;
context = train_stage';
y = smooth_vector(smooth,diff_map)';
SNR = 1;
load("..\data\paths\paths.mat");
save(fullfile('C:/Users/hadas-stud-group2/Documents/GitHub/neural_cstg/GaligoolAngel/data/inputs', datafile_name), 'y', ...
    'features', 'SNR','context');
output = y;
end