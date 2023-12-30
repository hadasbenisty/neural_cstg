
clear;
%filename = '../4458/02_03_19/data.mat';
filename = 'D:\flavorsProject\data\4458\02_03_19\data.mat'
load(filename);
X=[];outcome_label=[];time_win=[];behavioral_events = [];trials = [];
t = linspace(-4, 8, size(imagingData.samples,2));
trial_inds = 1:size(imagingData.samples,3);

for tind = 1:length(t)
    
    x = squeeze((imagingData.samples(:, tind, :)));
    X = cat(1, X, x');
    outcome_label = cat(1, outcome_label, BehaveData.success.indicatorPerTrial);
    time_win = cat(1, time_win, t(tind)*ones(size(BehaveData.success.indicatorPerTrial)));
    trials = cat(1, trials, trial_inds(:));
end
save([filename(1:end-4) 'python.mat'] ,'X','outcome_label','time_win','trials');
