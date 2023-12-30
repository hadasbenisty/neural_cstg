function plot_trials_traj(X, labels, clrs, leg)

X1 = X(:, :, labels>0);
labels = labels(labels > 0);

ul = unique(labels);

for c = 1:length(ul)
    k = (labels == ul(c));
    X2 = nanmean(X1(:, :, k), 3);
    plot3(X2(1, :), X2(2, :), X2(3, :), clrs(labels(find(k,1))));
    hold all;
end
for c = 1:length(ul)
    k = (labels == ul(c));
    X2 = nanmean(X1(:, :, k), 3);
    plot3(X2(1, 1), X2(2, 1), X2(3, 1),  'ks');
    hold all;
end

legend(leg, 'Location', 'Best');