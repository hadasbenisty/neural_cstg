function plot_trials_dots(X, labels, clrs, leg)

X1 = X(:, labels>0);
labels = labels(labels > 0);

ul = unique(labels);
for c = 1:length(ul)
    k = find(labels == ul(c), 1);
    plot3(X1(1, k), X1(2, k), X1(3, k), [clrs(labels(k)) '.']);
    hold all;
end
for k = 1:length(labels)
    plot3(X1(1, k), X1(2, k), X1(3, k),[clrs(labels(k)) '.']);
    hold all;
end
legend(leg, 'Location', 'Best');