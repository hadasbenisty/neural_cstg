function plot_indicative_fraction_flavor(datapath, resultspath, animals_names, figspath, excelpath)

animals_db = get_animals_list(datapath, animals_names);
fsample = 30;
params.slidingWinLen = 1;
params.slidingWinHop = 0.5;
tonetime = 4;
pvalue = 0.01;



% wins = wins+tonetime;


type1 = {'train','first','ongoing'};
type2 = {'random','batch'};
t1 = unique(type1);
t2 = unique(type2);

Mfrac = nan(3, 2, length(animals_names));
for animal_i = 1:length(animals_names)
    disp(animals_names{animal_i});
    datesList = animals_db{animal_i}.folder;
    I = []; frac = []; count = [];
    for ei = 1:length(datesList)
        frac(ei) = get_data(datapath, ...
            resultspath, animals_names{animal_i}, datesList{ei}, fsample, params, pvalue);
    end
    
    %     figure;tiledlayout('flow')
    for n1 = 1:length(t1)
        for n2 = 1:length(t2)
            indsession = strcmp(animals_db{animal_i}.type1, t1{n1}) & ...
                strcmp(animals_db{animal_i}.type2, t2{n2});
            indsession = indsession & animals_db{animal_i}.to_include == 2;
            x = frac(indsession);
            if isempty(x)
                Mfrac(n1, n2, animal_i) = nan;
            else
                Mfrac(n1, n2, animal_i) = nanmean(x, 2);
            end
        end
    end
    
end

figure;
barwitherr(nanstd(100*Mfrac, [], 3)'/sqrt(5), ...
            nanmean(100*Mfrac, 3)');
legend(t1)        
set(gca, 'XTickLabel', t2)       
ylabel('% stable indicative cells');
   

% mysave(gcf,fullfile(figspath, 'across_animals_stats',...
%     ['svm_single_cells_indicative_'  'stable_q_vs_all']))



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

function frac = get_data(datapath, ...
    resultspath, animal, folder, fsample, params, pvalue)

disp(folder);
Nnrnmsmax = 1e3;
frac = nan;

currfolder = fullfile(datapath, animal, folder);
datafile = fullfile(currfolder, 'data.mat');
if ~isfile(datafile)
    return;
end
load(datafile, 'imagingData');
resfile = fullfile(resultspath, ['svm_by_cell_' animal '_' folder '.mat']);
if ~isfile(resfile)
    
    return;
end
t = (0:size(imagingData.samples, 2)-1)/fsample;
[winstSec, winendSec] = getFixedWinsFine(round(t(end)), params.slidingWinLen, params.slidingWinHop);
tonetime = 4;
tmid = (winstSec+winendSec)/2 - tonetime;
Nnrns = size(imagingData.samples, 1);

load(resfile, 'acc');
count = nan(length(acc), 1);
I = nan(Nnrnmsmax, length(acc)-1);

for l = 1:length(acc)-1
    if isempty(acc{l})
        continue;
    end
    SEM = acc{l}.std/sqrt(acc{l}.trialsnum-1);
    ts = tinv(1-2*pvalue, (acc{l}.trialsnum)-1);      % T-Score
    isindicative = acc{l}.mean-ts*SEM > acc{l}.chance;
    [count(l), I(imagingData.roiNames(:,1), l)] = getIndicativeNrnsMean(isindicative, ...
        'consecutive', 2, 1, length(tmid));
    
    
end
frac = sum(nansum(I,2)>0)/Nnrns;

end