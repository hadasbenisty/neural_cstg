ii = 1;
aff_mat = [];
for knn = 1:100
    [aff_mat_current, sigma] = CalcInitAff2D( dR1, knn);
    aff_mat(:, :, ii) = aff_mat_current; 
    ii = ii + 1;
end

%% Affine Matrix Difference Analysis
aff_mat_diff = diff(aff_mat, [], 3); aff_mat_diff_mean = ...
    mean(aff_mat_diff, [1, 2]) ; aff_mat_diff_mean = ...
    squeeze(aff_mat_diff_mean);
avg_aff_mat = squeeze(mean(aff_mat, [1,2]));
aff_mat_diff_mean_normalized = aff_mat_diff_mean ./ avg_aff_mat(1:end-1);
aff_mat_mean_diff = figure;
x = 2:length(aff_mat_diff_mean_normalized) + 1;
x = x ./ (x - 1);
plot(x, aff_mat_diff_mean, 'DisplayName', ['Average Difference Between ' ...
    'Following Affine Matrices']);
title('Average Difference Between Following Affine Matrices');
xlabel('KNN next / KNN current')
ylabel('Average Difference Divided by the Average Value in the Affine Matrix')


