function [smoothed_vector] = smooth_vector(smooth_level,vector)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
smooth_func = @(vec) sgolayfilt(vec, ...
        2, smooth_level);
smoothed_vector = NaN(size(vector));
for dim = 1:size(vector, 2)
    smoothed_vector(:, dim) = smooth_func(vector(:, dim));
    smoothed_vector(:,dim) = (smoothed_vector(:, dim) - ...
        min(smoothed_vector(:, dim))) ./ ...
                                (max(smoothed_vector(:, dim)) - ...
                                min(smoothed_vector(:, dim))); 
end
end