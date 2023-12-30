function h = myerrorbar(x, m, l, u, c)
for k = 1:length(x)
    h = line([1 1]*x(k), m(k) + [-l(k) u(k)], 'Color',c, 'LineWidth', 1);
end
