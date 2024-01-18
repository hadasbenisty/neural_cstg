function d = calc_dprime_traj(xx, yy)

d = nan(size(xx, 2), 1);
if sum(yy) <=3
    return;
end
M1 = nanmean(xx(:,:,yy==1), 3);
M0 = nanmean(xx(:,:,yy==0), 3);


S1=[];S0=[];
for t = 1:size(xx, 2)
    S1(:, :, t) = cov(squeeze(xx(:,t,yy==1))');
    S0(:, :, t) = cov(squeeze(xx(:,t,yy==0))');
end

for t = 1:size(xx, 2)
    S = (S1(:, :, t) + S0(:, :, t))/2;
    d(t) = sqrt((M1(:, t)-M0(:, t))' * pinv(S) * (M1(:, t)-M0(:, t)));
end