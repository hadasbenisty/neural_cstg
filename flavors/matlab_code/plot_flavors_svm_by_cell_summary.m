function plot_flavors_svm_by_cell_summary(datapath, resultspath, animals_names, figspath, excelpath)

animals_db = get_animals_list(datapath, animals_names);
fsample = 30;
params.slidingWinLen = 1;
params.slidingWinHop = 0.5;
tonetime = 4;
pvalue = 0.01;

wins = [-4 -.1; 0 3; 3 20];
for i = 1:size(wins, 1)
    winsstrs{i, 1} = [num2str(wins(i, 1)) '_' num2str(wins(i, 2))];
end
for wini = 1:size(wins, 1)
    leg{wini} = sprintf('[%2.1f, %2.1f]', wins(wini, 1), wins(wini, 2));
end
% wins = wins+tonetime;

flavors = { 'quinines', 'sucroses', 'regulars', 'grains','fakes'};
pairs = nchoosek(1:length(flavors), 2);
for p = 1:size(pairs, 1)
    ttls{p} = [flavors{pairs(p, 1)} ' ' flavors{pairs(p, 2)}];
end
ttls{end+1} = 'success failure';
type1 = {'train','first','ongoing'};
type2 = {'random','batch'};
frac_indic_animals = nan(length(leg), length(ttls), length(type1), ...
    length(type2), length(animals_names));
frac_indic_animals_with_sf = nan(length(leg), length(ttls), length(type1), ...
    length(type2), length(animals_names));
count_indic_animals_with_sf = nan(length(leg), length(ttls), length(type1), ...
    length(type2), length(animals_names));
count_indic_animals = nan(length(leg), length(ttls), length(type1), ...
    length(type2), length(animals_names));
stability = nan(length(leg), length(ttls), length(type1), ...
    length(type2), length(animals_names));

for animal_i = 1:length(animals_names)
    disp(animals_names{animal_i});
    datesList = animals_db{animal_i}.folder;
    countindic_with_sf = [];
    fracindic = [];maxNrns = 0;
    IndicAll = nan(700, 3, 11, length(datesList));
    isgoodsesshion = false(length(datesList), 3);
    nrnsnames = [];minNrns = 1e3;
    resfileall = fullfile(resultspath, ['svm_cells_' animals_names{animal_i} '.mat']);
    if isfile(resfileall)
        load(resfileall, 'countindic_with_sf', 'fracindic_with_sf', 'countindic', ...
        'fracindic', 'IndicAll', 'isgoodsesshion' ,'maxNrns', 'minNrns')
    else
    for ei = 1:length(datesList)
        [t, imagingData, BehaveData, cellnames, accall, isindicative, isindicative_with_sf, ...
            indic_by_win, indic_by_win_with_sf, tmid, count_indic_win, count_indic_win_with_sf, ...
            frac_indic_win, frac_indic_win_with_sf] = ...
            get_data(datapath, resultspath, animals_names{animal_i}, ...
            datesList{ei}, wins, ...
            fsample, params, tonetime, pvalue, excelpath, ttls, winsstrs);
        if isempty(accall)
            continue;
        end
        if isempty(nrnsnames)
        nrnsnames = imagingData.roiNames(:,1);
        end
        maxNrns = max([imagingData.roiNames(:,1); maxNrns]);
        minNrns = min([max(imagingData.roiNames(:,1)); minNrns]);
        for l = 1:size(indic_by_win, 3)
            if ~isnan(indic_by_win(1, 1, l))
                fracindic(:, l, ei) = squeeze(sum(indic_by_win(:, :, l)>0.05))/size(indic_by_win, 1);
                countindic(:, l, ei) = squeeze(sum(indic_by_win(:, :, l)>0.05));
                
                
                IndicAll(imagingData.roiNames(:,1), :, l, ei) = indic_by_win(:, :, l);
                isgoodsesshion(ei, l) = true;
                
            else
                fracindic(:, l, ei) = nan(size(indic_by_win, 2), 1);
                countindic(:, l, ei) = nan(size(indic_by_win, 2), 1);
            end
            if ~isnan(indic_by_win(1, 1, l))
                countindic_with_sf(:, l, ei) = squeeze(sum(indic_by_win_with_sf(:, :, l)>0.05));
                fracindic_with_sf(:, l, ei) = squeeze(sum(indic_by_win_with_sf(:, :, l)>0.05))/size(indic_by_win_with_sf, 1);
            else
                fracindic_with_sf(:, l, ei) = nan(size(indic_by_win_with_sf, 2), 1);
                countindic_with_sf(:, l, ei) = nan(size(indic_by_win_with_sf, 2), 1);
            end
        end
        
    end
    save(resfileall, 'countindic_with_sf', 'fracindic_with_sf', 'countindic', ...
        'fracindic', 'IndicAll', 'isgoodsesshion' ,'maxNrns', 'minNrns');
    end
    
    indtest = find(contains(ttls, 'quinines'));
    for i = 1:length(indtest)
        x = squeeze(IndicAll(:, 3, indtest(i), :));
        if any(nansum(x>0.05)>20)
        selind = find(nansum(x>0.05)>20);
        selind = selind(end);
        break;
        end
    end
    
    IndicAll1 = IndicAll(1:minNrns, :, :, :);
    
    [~, ic] = sort(IndicAll1(:, 3, indtest(i), (selind)));
    %% plot indiv nrns through time
    for ei = 1:length(animals_db{animal_i}.type1)
        explabel{ei} = [animals_db{animal_i}.type1{ei}(1:2) ' ' animals_db{animal_i}.type2{ei}(1:2)];
    end
%     figure;k=1;
%     for wini = 1:3
%         for l = 1:size(isgoodsesshion, 2)
%             x = squeeze(IndicAll1(:, wini, l, :));
%             
%             
%             if all(isnan(x(:)))
%                 k=k+1;
%                 continue;                
%             end
%             subplot(3, size(isgoodsesshion, 2), k);            
%             k=k+1;
%             xx = x(:, sum(isnan(x))~=minNrns);
%             [~, ic] = sort(xx(:, end));
%             imagesc(x(ic, sum(isnan(x))~=minNrns), [0 1])
%             clrmp = [0 0 0; 1 1 1];
%             set(gca, 'XTick', 1:sum(sum(isnan(x))~=minNrns));set(gca, 'XTickLabel', explabel(sum(isnan(x))~=minNrns));  set(gca, 'XTickLabelRotation', 90);
%             set(gca, 'YTickLabel', []);
%             title(ttls{l})
% 
%         end
%     end
%     setfigbig;colormap jet;
%     suptitle(animals_names{animal_i});
%     mysave(gcf, fullfile(figspath, 'per_animal_stats', ['svm_cells_' animals_names{animal_i}]));
    t1 = unique(type1);
    t2 = unique(type2);
    %     figure;tiledlayout('flow')
    for n1 = 1:length(t1)
        for n2 = 1:length(t2)
            indsession = strcmp(animals_db{animal_i}.type1, t1{n1}) & ...
                strcmp(animals_db{animal_i}.type2, t2{n2});
            indsession = indsession & animals_db{animal_i}.to_include == 2;
            x = fracindic(:, :, indsession);
            Mfrac = nanmean(x, 3);
            Nfrac = sum(~isnan(x(1, :, :)),3);
            if all(Nfrac == 0)
                continue;
            end
            %             S =  nanstd(x, [], 3)./sqrt(N-1);
            frac_indic_animals(:, :, n1, n2, animal_i) = Mfrac;
            frac_indic_animals_with_sf(:, :, n1, n2, animal_i) = nanmean(fracindic_with_sf(:, :, indsession), 3);
            count_indic_animals(:, :, n1, n2, animal_i) = nanmean(countindic(:, :, indsession), 3);
            count_indic_animals_with_sf(:, :, n1, n2, animal_i) = nanmean(countindic_with_sf(:, :, indsession), 3);
            stability(:, :,n1, n2, animal_i) = sum(squeeze(nansum(IndicAll1(:, :, :, indsession),4))/sum(indsession) > 0.5)/minNrns;
            
            if all(isnan(Mfrac(:)))
                continue;
            end
            %             nexttile;
            %             inds = find(~isnan(M(1,:)));
            %             if any(N < 1)
            %                 bar(M(:, inds));
            %             else
            %                 barwitherr(S(:, inds), M(:, inds))
            %             end
            %             legend(ttls(inds), 'Location', 'Best');
            %             xlabel('Time window');set(gca, 'XTickLabel', leg);
            %             set(gca, 'XTickLabelRotation', 45);ttl = [t1{n1} ' ' t2{n2}];
            %             title(ttl);
        end
    end
%         setfigbig;suptitle(animals_names{animal_i});setfigbig
%         ttl1 = ttl;
%         ttl1(ttl1==' ') = '_';
%         mysave(gcf,fullfile(figspath, 'per_animal_stats', ...
%             [animals_names{animal_i} '_svm_single_cell_count'  ]));
%     close all;
end
winsstrs1 = winsstrs;
for k=1:length(winsstrs1)
    winsstrs1{k}(winsstrs1{k} == '_') = ' ';
end
 t1 = unique(type1);
    t2 = unique(type2);
figure;i=1;
for n1 = 1:length(t1)-1
    for n2 = 1:length(t2)
       
        sttl = [t1{n1} ' ' t2{n2}];
        subplot(length(t1)-1, length(t2), i);
        barwitherr(nanstd(100*stability(:, end, n1, n2, :), [], 5)/sqrt(5), ...
            nanmean(100*stability(:, end, n1, n2, :), 5));
        
        set(gca, 'XTickLabel', winsstrs1);
        title(sttl);i=i+1;
        ylabel('% stable indicative cells');
    end
end
  mysave(gcf,fullfile(figspath, 'across_animals_stats',...
            ['svm_single_cells_indicative_'  'stable_q_vs_all']))
        
        
        
Mfrac = nanmean(frac_indic_animals, 5);
Sfrac = nanstd(frac_indic_animals, [], 5);
Nfrac = squeeze(sum(~isnan(frac_indic_animals(1,:,:,:,:)),5));
Mcount_with_sf = nanmean(count_indic_animals_with_sf, 5);
Mcount = nanmean(count_indic_animals, 5);
Mfrac_with_sf = nanmean(frac_indic_animals_with_sf, 5);
maxMMfrac = max(max(max(Mfrac(:, 11, :, :))));
M00 = Mfrac/maxMMfrac;
for n1 = 1:length(t1)
    for n2 = 1:length(t2)
        sttl = [t1{n1} ' ' t2{n2}];
        if all(Nfrac(:, n1, n2) == 0)
            continue;
        end
        
        for l = 1:size(Nfrac, 1)
            if Nfrac(l, n1, n2) == 0
                continue;
            end
            figure;
            
            barwitherr(Sfrac(:, l, n1, n2)/sqrt(Nfrac(l, n1, n2)-1), ...
                Mfrac(:, l, n1, n2));
            title([sttl ' ' ttls{l} ' n=' num2str(Nfrac(l, n1, n2))]);ylim([0 .5])
            xlabel('Time window');set(gca, 'XTickLabel', leg);
            set(gca, 'XTickLabelRotation', 45);
            ylabel('Fraction of Cells');
            ttl1 = sttl;
            ttl1(ttl1==' ') = '_';
            ttl2 = ttls{l};
            ttl2(ttl2==' ') = '_';
            mysave(gcf,fullfile(figspath, 'across_animals_stats',...
                ['svm_single_cells_indicative_' ttl1 '_' ttl2]))
        end
        A = Mfrac(:, :, n1, n2);
        I = Mfrac_with_sf(:, :, n1, n2);
        for win = size(A, 1)
            
            for l = 1:size(A, 2)-1
                if isnan(A(win, l))
                    continue;
                end
                figure;
                venn(A(win, [l, size(A, 2)]),I(win, l));
                title(['Time win: ' leg{win} 'sec Intersect=' num2str(I(win, l)*100) '%']);
                l1=legend({[ttls{l} '=' num2str(A(win, l)*100) '%'] ...
                    ['suc/fail=' num2str(A(win, end)*100) '%']}, ...
                    'Location', 'Best');
                l1.EdgeColor='none';
                a = gca;
                a.XTickLabel = '';a.YTickLabel = '';
                ttl1 = sttl;
                ttl1(ttl1==' ') = '_';
                ttl2 = ttls{l};
                ttl2(ttl2==' ') = '_';
%                 P = get(gcf, 'Position');
%                 set(gcf, 'Position', [P(1:2) P(3:4)*M00(win, 11, n1, n2)]);
                mysave(gcf,fullfile(figspath, 'across_animals_stats',...
                    ['venn_svm_single_cells_indicative_' ttl1 '_' ttl2 '_win' num2str(win)]))
                
                close all;
            end
        end
        
        
        
    end
    
    
end

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

function [t, imagingData, BehaveData, cellnames, accall, isindicative, isindicative_with_sf, ...
    indic_by_win, indic_by_win_with_sf,...
    tmid, count_indic_win, count_indic_win_with_sf, frac_indic_win, frac_indic_win_with_sf] = get_data(datapath, ...
    resultspath, animal, folder, wins, fsample, params, tonetime, pvalue, excelpath, ttls, winsstrs)
cellnames = [];
accall = [];
isindicative = [];
isindicative_with_sf = [];
frac_indic_win = [];
frac_indic_win_with_sf = [];

indic_by_win = [];
indic_by_win_with_sf = [];

tmid = [];
count_indic_win = [];
count_indic_win_with_sf = [];

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
isindicative_with_sf = nan(Nnrns, length(tmid), length(acc));
indic_by_win = nan(Nnrns, size(wins, 1), length(acc));
indic_by_win_with_sf = nan(Nnrns, size(wins, 1), length(acc));
count_indic_win = nan(size(wins, 1), length(acc));
count_indic_win_with_sf = nan(size(wins, 1), length(acc));
frac_indic_win = nan(size(wins, 1), length(acc));
frac_indic_win_with_sf = nan(size(wins, 1), length(acc));
for l = 1:length(acc)
    if isempty(acc{l})
        continue;
    end
    list_indic_by_win(cellnames, excelpath, animal, folder, ...
        indic_by_win, ttls, winsstrs);
    accall(:, :, l) = acc{l}.mean - acc{l}.chance;
    SEM = acc{l}.std/sqrt(acc{l}.trialsnum-1);               % Standard Error
    ts = tinv(1-2*pvalue, (acc{l}.trialsnum)-1);      % T-Score
    isindicative(:, :, l) = acc{l}.mean-ts*SEM > acc{l}.chance;
    for win_i = 1:size(wins, 1)
        indic_by_win(:, win_i, l) = ...
            squeeze(nansum(isindicative(1:Nnrns, ...
            tmid>=wins(win_i, 1) & tmid<= wins(win_i, 2), l), 2))/...
            sum(tmid>=wins(win_i, 1) & tmid<= wins(win_i, 2));
    end
    
    
    count_indic_win(:, l) = squeeze(nansum(indic_by_win(:, :, l)>0.5, 1));
    
    frac_indic_win(:, l) = count_indic_win(:, l)/Nnrns;
end

for l = 1:size(indic_by_win_with_sf, 3)
    
    A = indic_by_win(:, :, l);
    B = indic_by_win(:, :, end);
    C = A;
    C(A==1 & B == 1) = 1;
    C(A==0 | B == 0) = 0;
    indic_by_win_with_sf(:, :, l) = C;
    count_indic_win_with_sf(:, l) = squeeze(nansum(indic_by_win_with_sf(:, :, l)>0.5, 1));
    
    frac_indic_win_with_sf(:, l) = count_indic_win_with_sf(:, l)/Nnrns;
end

end