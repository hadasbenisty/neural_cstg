function plot_flavors_dprime(datapath, resultspath, animals_names, figspath)

animals_db = get_animals_list(datapath, animals_names);
fsample = 30;
tonetime = 4;
flavors = { 'quinines', 'sucroses', 'regulars', 'grains','fakes'};
pairs = nchoosek(1:length(flavors), 2);
for p = 1:size(pairs, 1)
    ttls{p} = [flavors{pairs(p, 1)} ' ' flavors{pairs(p, 2)}];
end
ttls{end+1} = 'success failure';
plot_per_exp = false;
type1 = {'train','first','ongoing'};
type2 = {'random','batch'};
dprime_animals = nan(1000, size(pairs, 1)+1, length(type1), ...
    length(type2), length(animals_names));

for animal_i = 1:length(animals_names)
    disp(animals_names{animal_i});
    datesList = animals_db{animal_i}.folder;
    %     type1 = animals_db{animal_i}.type1;
    %     type2 = animals_db{animal_i}.type2;
    dprimeall = nan(1000, size(pairs, 1)+1, length(datesList));
    
    for ei = 1:length(datesList)
        disp(datesList{ei});
        currfolder = fullfile(datapath, animals_names{animal_i}, datesList{ei});
        datafile = fullfile(currfolder, 'data.mat');
        if ~isfile(datafile)
            disp([datafile ' missing']);
            continue;
        end
        resfile = fullfile(resultspath, ['dprime_trajectories_' animals_names{animal_i} '_' datesList{ei} '.mat']);
        if ~isfile(resfile)
            continue;
        end
        load(resfile, 'dprime');
        for l = 1:length(dprime)
            if isempty(dprime{l})
                continue;
            end
            tmax = length(dprime{l}.dprime);
            dprimeall(1:tmax, l, ei) = dprime{l}.dprime;
        end
        
        
        t = (0:size(dprimeall, 1)-1)/fsample - tonetime;
        if plot_per_exp
            figure;tiledlayout('flow')
            for l = 1:length(dprime)
                if isempty(dprime{l})
                    continue;
                end
                nexttile;
                m = dprime{l}.dprime;
                plot(t(1:length(m)), m);
                title(dprime{l}.labels);
                xlabel('Time [sec]');
                ylabel('Sensitivity Index');
                
                line([0 0], get(gca, 'YLim'), 'LineStyle', '-.', ...
                    'Color', 'k');axis tight;
                
                
            end
            mysave(gcf,fullfile(figspath, 'per_experiment_stats', [animals_names{animal_i} '_' datesList{ei} '_dprime_'   ]));
        end
    end
    
    t1 = unique(type1);
    t2 = unique(type2);
    
    for n1 = 1:length(t1)
        for n2 = 1:length(t2)
            ind = strcmp(animals_db{animal_i}.type1, t1{n1}) & ...
                strcmp(animals_db{animal_i}.type2, t2{n2});
            ind = ind & animals_db{animal_i}.to_include == 2;
            if all(~ind)
                continue;
            end
            x = permute(dprimeall(1, :, ind), [2 3 1]);
            if all(isnan(x(:)))
                continue;
            end
            n = sum(~isnan(x'), 1);
            ttl = [t1{n1} ' ' t2{n2}];
            m = squeeze(nanmean(dprimeall(:, :, ind), 3));
            dprime_animals(:, :, n1, n2, animal_i) = m;
            s = bsxfun(@rdivide, squeeze(nanstd(dprimeall(:, :, ind), [], 3)), sqrt(max(n-1, 1)));
            if plot_per_exp
                figure;tiledlayout('flow')
                for l = 1:size(m, 2)
                    if isnan(m(1, l))
                        continue;
                    end
                    nexttile;
                    
                    shadedErrorBar(t, m(:, l), s(:, l));
                    title([ttls{l} ' n=' num2str(n(l))]);axis tight;
                    
                    line([0 0], get(gca, 'YLim'), 'LineStyle', '-.', ...
                        'Color', 'k');
                    ylabel('Sensitivity Index');xlabel('Time [sec]');
                end
                suptitle(ttl);setfigbig
                ttl1 = ttl;
                ttl1(ttl1==' ') = '_';
                mysave(gcf,fullfile(figspath, 'per_animal_stats', [animals_names{animal_i} '_dprime_'  '_' ttl1 ]));
            end
        end
    end
    close all;
end

M = nanmean(dprime_animals, 5);
S = nanstd(dprime_animals, [], 5);
N = squeeze(sum(~isnan(dprime_animals(1,:,:,:,:)),5));

for n1 = 1:length(t1)
    for n2 = 1:length(t2)
        sttl = [t1{n1} ' ' t2{n2}];
        if all(N(:, n1, n2) == 0)
            continue;
        end
        
        for l = 1:size(N, 1)
            if N(l, n1, n2) == 0
                continue;
            end
            figure;
            
            shadedErrorBar(t, M(:, l, n1, n2), ...
                S(:, l, n1, n2)/sqrt(N(l, n1, n2)-1));
            title([sttl ' ' ttls{l} ' n=' num2str(N(l, n1, n2))]);axis tight;
            
            line([0 0], get(gca, 'YLim'), 'LineStyle', '-.', ...
                'Color', 'k');
            ylabel('Sensitivity Index');xlabel('Time [sec]');
            
            
            
            
            ttl1 = sttl;
            ttl1(ttl1==' ') = '_';
            ttl2 = ttls{l};
            ttl2(ttl2==' ') = '_';
            mysave(gcf,fullfile(figspath, 'across_animals_stats',...
                ['dprime_population_' ttl1 '_' ttl2]))
            
        end
        
    end
    
    
end
end
