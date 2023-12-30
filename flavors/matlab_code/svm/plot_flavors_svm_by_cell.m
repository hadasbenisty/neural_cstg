function plot_flavors_svm_by_cell(datapath, resultspath, animals_names, figspath, excelpath)

animals_db = get_animals_list(datapath, animals_names);
fsample = 30;
params.slidingWinLen = 1;
params.slidingWinHop = 0.5;
tonetime = 4;
pvalue = 0.01;
wins = [-3.5 -.1; 0 3; 3 20];
for i = 1:size(wins, 1)
    winsstrs{i, 1} = [num2str(wins(i, 1)) '_' num2str(wins(i, 2))];
end
wins = wins+tonetime;

flavors = {'failure' 'quinines', 'sucroses', 'regulars', 'grains','fakes'};
pairs = nchoosek(1:length(flavors), 2);
for p = 1:size(pairs, 1)
    ttls{p} = [flavors{pairs(p, 1)} ' ' flavors{pairs(p, 2)}];
end
ttls{end+1} = 'success failure';
for animal_i = 1:length(animals_names)
    disp(animals_names{animal_i});
    datesList = animals_db{animal_i}.folder;
    for ei = 1:length(datesList)
        [t, imagingData, BehaveData, cellnames, accall, isindicative, indic_by_win, tmid, count_indic_win] = ...
            get_data(datapath, resultspath, animals_names{animal_i}, ...
            datesList{ei}, wins, ...
            fsample, params, tonetime, pvalue);
        if isempty(accall)
            continue;
        end
        
        
        
        %         plot_raw_indic(isindicative, tmid, indic_by_win, accall, ttls, [animals_names{animal_i} ' ' animals_db{animal_i}.type1{ei}], ...
        %             animals_names{animal_i}, datesList{ei}, figspath, wins-tonetime);
        %         plot_count_indic(figspath, count_indic_win, ttls, animals_names{animal_i}, ...
        %             datesList{ei}, wins-tonetime);
        %         plot_score_indicative(figspath, animals_names{animal_i}, datesList{ei}, ...
        %             count_indic_win, indic_by_win, wins, ttls);
        plot_activity_per_nrn(figspath, t-tonetime, imagingData, BehaveData, flavors, indic_by_win, ...
            animals_names{animal_i}, datesList{ei}, wins)
        
        
        close all
        
        
        
    end
    
end
end
function plot_activity_per_nrn(figspath, t, imagingData, BehaveData, flavors, indic_by_win, animal, expname, wins)
clrs = 'rgmcky';

inds = find(~isnan(squeeze(indic_by_win(1,1,:))));
for l = 1:length(inds)
    if l == length(inds)
        if ~isfield(BehaveData, 'success')
            continue;
        end
        for win_i = 1:size(wins, 1)
            nrns2plot = find(indic_by_win(:, win_i, inds(l))>0.3);
            for nrni = 1:length(nrns2plot)
                figfile = fullfile(figspath, 'individual_cells', 'fig', ...
                    [ animal '_' expname '_cell' ...
                    num2str(nrns2plot(nrni)) '_s_f.fig']);
                if isfile(figfile)
                    continue;
                end
                X = squeeze(imagingData.samples(nrns2plot(nrni), :, :));
                Y = BehaveData.success.indicatorPerTrial;
                M = [];
                S = [];
                M(:, 2) = nanmean(X(:, Y== 1), 2);
                N = sum(Y==1);
                S(:, 2) = nanstd(X(:, Y== 1), [], 2)/sqrt(N-1);
                M(:, 1) = nanmean(X(:, Y== 0), 2);
                N = sum(Y==0);
                S(:, 1) = nanstd(X(:, Y== 0), [], 2)/sqrt(N-1);
                
                
                
                if isempty(Y)
                    continue;
                end
                L = {'failure' 'success' };
                [Y, ic] = sort(Y);
                X = X(:, ic);
                figure;subplot(2,1,1);imagesc(t, 1:length(Y), X', [0 2])
                xlim([t(20) t(end)])
                set(gca, 'YTick', 1:4:length(Y));
                set(gca, 'YTickLabel', L(1+Y(1:4:end)));
                line([0 0], get(gca, 'YLim'), 'Color', 'w');
                c = colorbar;
                c.Label.String = '\DeltaF/F';
                c.Position = [0.8869    0.5857    0.0381    0.3381];
                xlabel('Time [sec]');
                colormap jet;
                subplot(2, 1, 2);
                
                for ci = 1:size(M, 2)
                    if isnan(M(1, ci))
                        continue;
                    end
                    shadedErrorBar(t, M(:, ci), S(:, ci), 'lineprops', clrs(ci))
                    hold all;
                    
                end
                c=get(gca, 'Children');
                xlim([t(20) t(end)])
                line([0 0], get(gca, 'YLim'));
                legend(c(end-1:-4:1), L);
                
                xlabel('Time [sec]');
                ylabel('\Delta F/F');title('Mean Activity');
                mysave(gcf, fullfile(figspath, 'individual_cells', ...
                    [ animal '_' expname '_cell' ...
                    num2str(nrns2plot(nrni)) '_s_f' ]))
                
                close all;
            end
        end
    else
        for win_i = 1:size(wins, 1)
            [~, ic1] = sort(indic_by_win(:, win_i, inds(l)), 'descend');
            
            nrns2plot = find(indic_by_win(ic1, win_i, inds(l))>0.3);
            
            for nrni = 1:length(nrns2plot)
                figfile = fullfile(figspath, 'individual_cells', 'fig', ...
                    [ animal '_' expname '_cell' ...
                    num2str(ic1(nrns2plot(nrni))) '.fig']);
                if isfile(figfile)
                    continue;
                end
                X = squeeze(imagingData.samples(ic1(nrns2plot(nrni)), :, :));
                
                if isfield(BehaveData, 'failure')
                  Y = BehaveData.failure.indicatorPerTrial;
                else
                    Y = 1-BehaveData.success.indicatorPerTrial;
                end
                
                M(:, 1) = nanmean(X(:, Y == 1), 2);
                N = sum(Y);
                S(:, 1) = nanstd(X(:, Y == 1), [], 2)/sqrt(N-1);
                
                        
                for ci = 2:length(flavors)
                    if isfield(BehaveData, flavors{ci})
                        Y(BehaveData.(flavors{ci}).indicatorPerTrial == 1) = ci;
                        M(:, ci) = nanmean(X(:, BehaveData.(flavors{ci}).indicatorPerTrial == 1), 2);
                        N = sum(BehaveData.(flavors{ci}).indicatorPerTrial);
                        S(:, ci) = nanstd(X(:, BehaveData.(flavors{ci}).indicatorPerTrial == 1), [], 2)/sqrt(N-1);
                    else
                        M(:, ci) = nan(size(X, 1), 1);
                        S(:, ci) = nan(size(X, 1), 1);
                    end
                end
                X = X(:, Y ~= 0);
                Y = Y(Y ~= 0);
                if isempty(Y)
                    continue;
                end
                [Y, ic] = sort(Y);
                X = X(:, ic);
                figure;subplot(2,1,1);imagesc(t, 1:length(Y), X', [0 2])
                xlim([t(20) t(end)])
                set(gca, 'YTick', 1:4:length(Y));
                set(gca, 'YTickLabel', flavors(Y(1:4:end)));
                line([0 0], get(gca, 'YLim'), 'Color', 'w');
                c = colorbar;
                c.Label.String = '\DeltaF/F';
                c.Position = [0.8869    0.5857    0.0381    0.3381];
                xlabel('Time [sec]');
                colormap jet;
                subplot(2, 1, 2);
                leg = [];
                for ci = 1:size(M, 2)
                    if isnan(M(1, ci))
                        continue;
                    end
                    shadedErrorBar(t, M(:, ci), S(:, ci), 'lineprops', clrs(ci))
                    hold all;
                    leg{end+1} = flavors{ci};
                end
                c=get(gca, 'Children');
                xlim([t(20) t(end)])
                line([0 0], get(gca, 'YLim'));
                legend(c(end-1:-4:1), leg);
                
                xlabel('Time [sec]');
                ylabel('\Delta F/F');title('Mean Activity');
                mysave(gcf, fullfile(figspath, 'individual_cells', ...
                    [ animal '_' expname '_cell' ...
                    num2str(ic1(nrns2plot(nrni))) ]))
                
                close all;
            end
        end
    end
end
end
function plot_raw_indic(isindicative, tmid, indic_by_win, accall, ttls, bigttl, ...
    animal, expname, figspath, wins)
inds = find(~isnan(accall(1,1,:)));
isindicative = isindicative(:, :, inds);
indic_by_win = indic_by_win(:, :, inds);
accall = accall(:, :, inds);
ttls = ttls(inds);
figure;l=1;
for l1 = 1:size(accall, 3)
    [~, ic] = sort(sum(accall(:, :, l1), 2));
    for l2 = 1:size(accall, 3)
        subplot(size(accall, 3), size(accall, 3), l);
        m = accall(:, :, l2);
        Nnrns = size(m, 1);
        imagesc(tmid, 1:Nnrns, m(ic, :), [-.5 .5]);
        title([ttls{l2} ' by ' ttls{l1}]);
        xlabel('Time [sec]');
        ylabel('nrns');
        
        line([0 0], get(gca, 'YLim'), 'LineStyle', '-.', ...
            'Color', 'k');axis tight;
        c=colorbar;
        c.Label.String='\Delta Accuracy';
        l=l+1;
    end
end
colormap(redblue);
mysave(gcf,fullfile(figspath, 'per_experiment_stats', ...
    [animal '_' expname '_svm_acc_single_cell'  ]))
figure;l=1;
for l1 = 1:size(accall, 3)
    [~, ic] = sort(sum(accall(:, :, l1), 2));
    for l2 = 1:size(accall, 3)
        subplot(size(accall, 3), size(accall, 3), l);
        
        imagesc(tmid, 1:Nnrns, isindicative(ic, :, l2))
        xlabel('Time [sec]');
        ylabel('nrns');
        title([ttls{l2} ' by ' ttls{l1}]);
        line([0 0], get(gca, 'YLim'), 'LineStyle', '-.', ...
            'Color', 'k');axis tight;
        c=colorbar;
        c.Label.String='Indicative';
        l=l+1;
    end
end
mysave(gcf,fullfile(figspath, 'per_experiment_stats', ...
    [animal '_' expname '_svm_indic_single_cell'  ]))

Np = size(accall, 3);
if Np == 1
    return;
end
pairs = nchoosek(1:Np, 2);

for wini = 1:size(indic_by_win, 2)
    figure;tiledlayout('flow')
    for pii = 1:size(pairs, 1)
        nexttile;
        plot(indic_by_win(:, wini, pairs(pii, 1)),...
            indic_by_win(:, wini, pairs(pii, 2)), 'k.');
        xlabel(ttls{pairs(pii, 1)});
        ylabel(ttls{pairs(pii, 2)});
        
    end
    suptitle(['Indicative Scores [' num2str(wins(wini, 1)) ', ' ...
        num2str(wins(wini, 2)) ']sec']);
    mysave(gcf,fullfile(figspath, 'per_experiment_stats', ...
        [animal '_' expname '_svm_scatter_single_cell_win' num2str(wini)  ]))
    
end

end
function plot_score_indicative(figspath, animal, expname, count_indic_win, indic_by_win, wins, ttls)
inds = find(sum(count_indic_win)>0);
figure;l=1;
for win_i = 1:size(wins, 1)
    for n1 = 1:length(inds)
        [~, ic] = sort(indic_by_win(:, win_i, inds(n1)));
        subplot(size(wins, 1), length(inds), l);
        imagesc(squeeze(indic_by_win(ic, win_i, inds)));
        set(gca, 'XTick', 1:length(inds));
        set(gca, 'XTickLabel', ttls(inds));
        set(gca, 'XTickLabelRotation', 90);
        title(['Win. ' num2str(win_i) ' by ' ttls{inds(n1)}]);
        ylabel('Cells');
        c = colorbar;
        c.Label.String = 'Indic Score';
        l=l+1;
    end
end
setfigbig;colormap(jet);
mysave(gcf,fullfile(figspath, 'per_experiment_stats', ...
    [animal '_' expname 'svm_score_indicative_wins' ]))


end
function plot_count_indic(figspath, count_indic_win, ttls, animal, expname, wins)
figure;
inds = find(sum(count_indic_win)>0);
bar(count_indic_win(:, inds));
legend(ttls(inds), 'Location', 'BestOutside')
xlabel('time window');
for wini = 1:size(wins, 1)
    leg{wini} = sprintf('[%2.1f, %2.1f]', wins(wini, 1), wins(wini, 2));
end
set(gca, 'XTickLabel', leg);
set(gca, 'XTickLabelRotation', 45);
ylabel('Fraction of Cells');
mysave(gcf,fullfile(figspath, 'per_experiment_stats', ...
    [animal '_' expname '_svm_count_single_cell'  ]))


end


function [t, imagingData, BehaveData, cellnames, accall, isindicative, ...
    indic_by_win, tmid, count_indic_win] = get_data(datapath, ...
    resultspath, animal, folder, wins, fsample, params, tonetime, pvalue)
cellnames = [];
accall = [];
isindicative = [];

indic_by_win = [];

tmid = [];
count_indic_win = [];

imagingData = [];
BehaveData = [];
t = [];
disp(folder);
currfolder = fullfile(datapath, animal, folder);
datafile = fullfile(currfolder, 'data.mat');
if ~isfile(datafile)
    return;
end
resfile = fullfile(resultspath, ['svm_by_cell_' animal '_' folder '.mat']);
if ~isfile(resfile)
    return;
end
load(datafile, 'imagingData', 'BehaveData');
cellnames = imagingData.roiNames(:, 1);
t = (0:size(imagingData.samples, 2)-1)/fsample;
[winstSec, winendSec] = getFixedWinsFine(round(t(end)), params.slidingWinLen, params.slidingWinHop);

tmid = (winstSec+winendSec)/2 - tonetime;
Nnrns = size(imagingData.samples, 1);

load(resfile, 'acc');
accall = nan(Nnrns, length(tmid), length(acc));
isindicative = nan(Nnrns, length(tmid), length(acc));
indic_by_win = nan(Nnrns, size(wins, 1), length(acc));
count_indic_win = nan(size(wins, 1), length(acc));
frac_indic_win = nan(size(wins, 1), length(acc));
for l = 1:length(acc)
    if isempty(acc{l})
        continue;
    end
     accall(:, :, l) = acc{l}.mean - acc{l}.chance;
    SEM = acc{l}.std/sqrt(acc{l}.trialsnum-1);               % Standard Error
    ts = tinv(1-2*pvalue, (acc{l}.trialsnum)-1);      % T-Score
    isindicative(:, :, l) = acc{l}.mean-ts*SEM > acc{l}.chance;
    [count, I] = getIndicativeNrnsMean(isindicative(:, :, l), ...
        'consecutive', 2, 0, 20);

    for win_i = 1:size(wins, 1)
        indic_by_win(:, win_i, l) = ...
            squeeze(nansum(isindicative(1:Nnrns, ...
            tmid>=wins(win_i, 1) & tmid<= wins(win_i, 2), l), 2))/...
            sum(tmid>=wins(win_i, 1) & tmid<= wins(win_i, 2));
    end
    
    
    count_indic_win(:, l) = squeeze(nansum(indic_by_win(:, :, l)>0.5, 1));
    
    frac_indic_win(:, l) = count_indic_win(:, l)/Nnrns;
end



end
