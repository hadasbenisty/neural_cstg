function plot_flavors_and_sf_svm_by_cell_summary(datapath, resultspath, animals_names, figspath, excelpath)

animals_db = get_animals_list(datapath, animals_names);
fsample = 30;
params.slidingWinLen = 1;
params.slidingWinHop = 0.5;
tonetime = 4;
pvalue = 0.01;

wins = [-4 -.1;0 3; 3 5; 5 20];
wins = [-3.5 -.1; 0 3; 3 20];

for i = 1:size(wins, 1)
    winsstrs{i, 1} = [num2str(wins(i, 1)) '_' num2str(wins(i, 2))];
end
for wini = 1:size(wins, 1)
    leg{wini} = sprintf('[%2.1f, %2.1f]', wins(wini, 1), wins(wini, 2));
end
% wins = wins+tonetime;



type1 = {'train','first','ongoing'};
type2 = {'random','batch'};

for animal_i = 1:length(animals_names)
    disp(animals_names{animal_i});
    datesList = animals_db{animal_i}.folder;
    
    resfilealloutc = fullfile(resultspath, ['svm_cells_' animals_names{animal_i} '_v2.mat']);
        resfileallflav = fullfile(resultspath, ['svm_cells_' animals_names{animal_i} '_q_vs_all.mat']);

       outc= load(resfilealloutc, 'countindic', ...
            'fracindic', 'del_acc_meanAll','IndicAll', 'minNrns');
     flav = load(resfileallflav, 'countindic', ...
            'fracindic', 'del_acc_meanAll','IndicAll', 'maxNrns', 'minNrns');

    
   minNrns = flav.minNrns;
     flav.IndicAll1 = flav.IndicAll(1:flav.minNrns, :, :);
    outc.IndicAll1 = outc.IndicAll(1:flav.minNrns, :, :);
    IndicAll1 = zeros(size(outc.IndicAll1));
    IndicAll1(outc.IndicAll1 == 1 & flav.IndicAll1 == 0) = 1;
    IndicAll1(outc.IndicAll1 == 0 & flav.IndicAll1 == 1) = 2;
    IndicAll1(outc.IndicAll1 == 1 & flav.IndicAll1 == 1) = 3;
    
    %% plot indiv nrns through time
    for ei = 1:length(animals_db{animal_i}.type1)
        explabel{ei} = [animals_db{animal_i}.type1{ei}(1:2) ' ' animals_db{animal_i}.type2{ei}(1:2)];
    end
    
    figure;
    for wini = 1:3
        x = squeeze(IndicAll1(:, wini, :));
        if all(isnan(x(:)))
            
            continue;
        end
        subplot(1, 3, wini);
        xx = x(:, sum(isnan(x))~=minNrns);
        sel = find(strcmp(explabel(sum(isnan(x))~=minNrns), 'on ba'));
        [~, ic] = sort(xx(:, sel(end)));
        
        subplot(1, 3, wini);
        imagesc(x(ic, sum(isnan(x))~=minNrns), [0 3])
        set(gca, 'XTick', 1:sum(sum(isnan(x))~=minNrns));set(gca, 'XTickLabel', explabel(sum(isnan(x))~=minNrns));  set(gca, 'XTickLabelRotation', 90);
        set(gca, 'YTickLabel', []);
        tt = winsstrs{wini};
        tt(tt=='_') = ' ';
        title(tt)
    end
    colormap jet;
%     suptitle(animals_names{animal_i});
%     mysave(gcf, fullfile(figspath, 'per_animal_stats', ['svm_cells_' animals_names{animal_i} '_suc_fail']));
    t1 = unique(type1);
    t2 = unique(type2);
    %     figure;tiledlayout('flow')
    for n1 = 1:length(t1)
        for n2 = 1:length(t2)
            indsession = strcmp(animals_db{animal_i}.type1, t1{n1}) & ...
                strcmp(animals_db{animal_i}.type2, t2{n2});
            indsession = indsession & animals_db{animal_i}.to_include == 2;
            x = sum(IndicAll1(:, :, indsession));
            Mfrac = nanmean(x, 2);
            Nfrac = sum(~isnan(x(1, :)),2);
            if all(Nfrac == 0)
                continue;
            end
            frac_indic_animals(:, n1, n2, animal_i) = Mfrac;
            count_indic_animals(:, n1, n2, animal_i) = nanmean(countindic(:, indsession), 2);
            stability(:, n1, n2, animal_i) = sum(squeeze(nansum(IndicAll1(:, :, indsession),3))/sum(indsession) > 0.5)/minNrns;
            
            if all(isnan(Mfrac(:)))
                continue;
            end
            
        end
    end
    
end

figure;i=1;
for n1 = 1:length(t1)
    for n2 = 1:length(t2)
       
        sttl = [t1{n1} ' ' t2{n2}];
        subplot(length(t1), length(t2), i);
        barwitherr(nanstd(100*stability(:, n1, n2, :), [], 4)/sqrt(5), ...
            nanmean(100*stability(:, n1, n2, :), 4));
        
        set(gca, 'XTickLabel', leg);
        set(gca, 'XTickLabelRotation',45);
        title(sttl);i=i+1;
        ylabel('% stable indicative cells');
    end
end
        
        Mfrac = nanmean(frac_indic_animals, 4);
Sfrac = nanstd(frac_indic_animals, [], 4);
Nfrac = squeeze(sum(~isnan(frac_indic_animals(1,:,:,:)),4));
Mcount = nanmean(count_indic_animals, 4);
maxMMfrac = max(max(max(Mfrac)));
figure;i=1;
for n1 = 1:length(t1)
    for n2 = 1:length(t2)
        sttl = [t1{n1} ' ' t2{n2}];
        
        
        
        subplot(3,2,i);
        
        barwitherr(100*Sfrac(:, n1, n2)/sqrt(max([Nfrac(n1, n2), 2])-1), ...
            100*Mfrac(:, n1, n2));
        title([sttl ' q vs all n=' num2str(Nfrac(n1, n2))]);ylim([0 50])
        xlabel('Time window');set(gca, 'XTickLabel', leg);
        set(gca, 'XTickLabelRotation', 45);
        ylabel('% Indicative');
        ttl1 = sttl;
        ttl1(ttl1==' ') = '_';
        ttl2 = 'suc fail';i=i+1;
       
        
        
        
    end
    
    
end
 mysave(gcf,fullfile(figspath, 'across_animals_stats',...
            ['svm_single_cells_indicative_' ttl1 '_' ttl2 ,'_v2']))
        
end
function list_indic_by_win(cellnames, excelpath, animal, expname, indic_by_win, ttls, winsstrs)
filename = fullfile(excelpath, [ animal '_' expname 'svm_single_cell_indicative_scores.xls']);
for nrni = 1:length(cellnames)
    cellnames1{nrni} = num2str(nrni);
end
inds = find(~isnan(squeeze(indic_by_win(1,1,:))));
for l = 1:length(inds)
    
    T = array2table(indic_by_win(:, :, inds(l)), 'RowNames', cellnames1, 'VariableNames', winsstrs);
    writetable(T, filename, 'Sheet', ttls{inds(l)}, 'WriteRowNames', true)
end
end

function [t, imagingData, BehaveData, cellnames, accall, isindicative, ...
    indic_by_win, tmid, count_indic_win, ...
    frac_indic_win, del_acc_mean] = get_data_q_vs_all(datapath, ...
    resultspath, animal, folder, wins, fsample, params, tonetime, pvalue, excelpath, ttls, winsstrs)
cellnames = [];
del_acc_mean=[];
accall = [];
isindicative = [];
frac_indic_win = [];
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
indic_by_win = nan(Nnrns, size(wins, 1));
del_acc_mean = nan(Nnrns, size(wins, 1));
accall = acc{end}.mean - acc{end}.chance;
SEM = acc{end}.std/sqrt(acc{end}.trialsnum-1);               % Standard Error
ts = tinv(1-2*pvalue, (acc{end}.trialsnum)-1);      % T-Score
isindicative = acc{end}.mean-ts*SEM > acc{end}.chance;
for win_i = 1:size(wins, 1)
    
     x = isindicative(1:Nnrns, ...
        tmid>=wins(win_i, 1) & tmid<= wins(win_i, 2));
    
    indic_by_win(:, win_i) = sum(x > 0, 2)>0;
    del_acc_mean(:, win_i) = mean(acc{end}.mean(1:Nnrns, ...
        tmid>=wins(win_i, 1) & tmid<= wins(win_i, 2)), 2) - acc{end}.chance;
end


count_indic_win = squeeze(nansum(indic_by_win, 1));
frac_indic_win = count_indic_win/Nnrns;




end