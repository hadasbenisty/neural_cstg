function dR = calc_Rdist(mC)




NN = size(mC, 3);
r=[];
for i = 1:NN
    e = sort(eig(mC(:, :, i)), 'descend');
    t = find(e < 0.01, 1)-1;
    if isempty(t)
        r(end+1) = length(e);
    else
        e = t;
    end
end
r = min(r);

dR = zeros(NN);
for i = 1:NN
    for j = i+1:NN
        dR(j, i) = SpsdDist(mC(:, :, i), mC(:, :, j), r);
        dR(i, j) = dR(j, i);
    end
end



end