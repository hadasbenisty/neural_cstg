function plot_flavors_svm_q_vs_all(datapath, resultspath, animals_names, figspath)

addpath(genpath('svm'));
animals_db = get_animals_list(datapath, animals_names);
fsample = 30;
params.slidingWinLen = 1;
params.slidingWinHop = 0.5;
tonetime = 4;
plot_per_animal = true;
flavors = {'sucroses', 'regulars', 'grains','fakes'};
qlabel = 'quinines';

type1 = {'train','first','ongoing'};
type2 = {'random','batch'};
plot_per_exp = true;
acc_animals = nan(100, length(type1), ...
    length(type2), length(animals_names));
for animal_i = 1:length(animals_names)
    disp(animals_names{animal_i});
    datesList = animals_db{animal_i}.folder;
    
    accall = nan(100, length(datesList));
    
    for ei = 1:length(datesList)
        disp(datesList{ei});
        currfolder = fullfile(datapath, animals_names{animal_i}, datesList{ei});
        datafile = fullfile(currfolder, 'data.mat');
        if ~isfile(datafile)
            continue;
        end
        resfile = fullfile(resultspath, ['svm_' animals_names{animal_i} '_' datesList{ei} '_q_vs_all.mat']);
        if ~isfile(resfile)
            continue;
        end
        load(datafile, 'imagingData');
        t = (0:size(imagingData.samples, 2)-1)/fsample;
        [winstSec, winendSec] = getFixedWinsFine(round(t(end)), params.slidingWinLen, params.slidingWinHop);
        tmid = (winstSec+winendSec)/2 - tonetime;
        
        load(resfile, 'acc');
        if plot_per_exp
            figure;
            
            
            tmax = length(acc.mean);
            accall(1:tmax, ei) = acc.mean - acc.chance;
            s = acc.std/sqrt(acc.trialsnum-1);
            shadedErrorBar(tmid, accall(1:tmax, ei), s);
            title(acc.labels);
            axis tight;ylim([-.5 .5])
            line(get(gca, 'XLim'), [0 0], 'LineStyle', '--', ...
                'Color', 'k');
            line([0 0], get(gca, 'YLim'), 'LineStyle', '-.', ...
                'Color', 'k');
            ylabel('\Delta Accuracy');xlabel('Time [sec]');
            
            
            
            
            
            mysave(gcf,fullfile(figspath, 'per_experiment_stats', [animals_names{animal_i} '_' datesList{ei} '_svm_q_vs_all'   ]));
        end
        
        
        tmax = length(acc.mean);
        accall(1:tmax, ei) = acc.mean - acc.chance;
        
    end
    
    t1 = unique(type1);
    t2 = unique(type2);
    
    for n1 = 1:length(t1)
        for n2 = 1:length(t2)
            ind = strcmp(animals_db{animal_i}.type1, t1{n1}) & ...
                strcmp(animals_db{animal_i}.type2, t2{n2});
            ind = ind & animals_db{animal_i}.to_include == 2;
            x = accall(1, ind);
            n = sum(~isnan(x'), 1);
            if all(n == 0)
                continue;
            end
            ttl = [t1{n1} ' ' t2{n2}];
            m = squeeze(nanmean(accall(:, ind), 2));
            acc_animals(:, n1, n2, animal_i) = m;
            s = bsxfun(@rdivide, squeeze(nanstd(accall(:, ind), [], 2)), sqrt(max(n-1, 1)));
            if plot_per_animal
                figure;
                
                if isnan(m(1))
                    continue;
                end
               
                
                shadedErrorBar(tmid, m(1:length(tmid)), s(1:length(tmid)));
                title(['quinine vs all n=' num2str(n)]);axis tight;ylim([-.5 .5])
                line(get(gca, 'XLim'), [0 0], 'LineStyle', '--', ...
                    'Color', 'k');
                line([0 0], get(gca, 'YLim'), 'LineStyle', '-.', ...
                    'Color', 'k');
                ylabel('\Delta Accuracy');xlabel('Time [sec]');
                
                suptitle(ttl);setfigbig
                ttl1 = ttl;
                ttl1(ttl1==' ') = '_';
                mysave(gcf,fullfile(figspath, 'per_animal_stats', [animals_names{animal_i} '_svm_population_' ttl1 '_svm_q_vs_all']))
            end
        end
    end
    
end
M = nanmean(acc_animals, 4);
S = nanstd(acc_animals, [], 4);
N = squeeze(sum(~isnan(acc_animals(1,:,:,:)),4));

for n1 = 1:length(t1)
    for n2 = 1:length(t2)
        sttl = [t1{n1} ' ' t2{n2}];
        if all(N(n1, n2) == 0)
            continue;
        end
        
            if N(n1, n2) == 0
                continue;
            end
            figure;
            
            shadedErrorBar(tmid, M(1:length(tmid), n1, n2), ...
                S(1:length(tmid), n1, n2)/sqrt(N(n1, n2)-1));
            title([sttl ' q vs all '  ' n=' num2str(N(n1, n2))]);axis tight;ylim([-.5 .5])
            line(get(gca, 'XLim'), [0 0], 'LineStyle', '--', ...
                'Color', 'k');
            line([0 0], get(gca, 'YLim'), 'LineStyle', '-.', ...
                'Color', 'k');
            ylabel('\Delta Accuracy');xlabel('Time [sec]');
            ttl1 = sttl;
            ttl1(ttl1==' ') = '_';
            
            mysave(gcf,fullfile(figspath, 'across_animals_stats',...
                ['svm_population_' ttl1 '_q_vs_all' ]))
        
        
    end
    
    
end
end

