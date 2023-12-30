function dPrime_analysis(datapath, resultspath, animals_names)

animals_db = get_animals_list(datapath, animals_names);
addpath(genpath('pca'));
flavors = { 'quinines', 'sucroses', 'regulars', 'grains','fakes'};
pairs = nchoosek(1:length(flavors), 2);


for animal_i = 1:length(animals_names)
    disp(animals_names{animal_i});
    datesList = animals_db{animal_i}.folder;
    
    
    for ei = 1:length(datesList)
        try
            disp(datesList{ei});
            currfolder = fullfile(datapath, animals_names{animal_i}, datesList{ei});
            datafile = fullfile(currfolder, 'data.mat');
            resfilepca = fullfile(resultspath, ['pca_trajectories_' animals_names{animal_i} '_' datesList{ei} '.mat']);
            
            if ~isfile(datafile) || ~isfile(resfilepca)
                continue;
            end
            resfile = fullfile(resultspath, ['dprime_trajectories_' animals_names{animal_i} '_' datesList{ei} '.mat']);
            if isfile(resfile)
                continue;
            end
            load(resfilepca, 'pcaTrajres');
            load(datafile, 'BehaveData');
            
            dprime = cell(size(pairs, 1)+1, 1);
            for pair_i = 1:size(pairs, 1)
                [X, Y] = get_data(pcaTrajres.projeff, BehaveData, flavors{pairs(pair_i, 1)}, ...
                    flavors{pairs(pair_i, 2)});
                if ~isempty(X) && ~isempty(Y)
                    dprime{pair_i}.dprime = getDprime(X,Y);
                    dprime{pair_i}.labels = [flavors{pairs(pair_i, 1)} ' ' flavors{pairs(pair_i, 2)}];
                    dprime{pair_i}.trialsnum = length(Y);
                    dprime{pair_i}.chance = sum(Y==1)/length(Y);
                    dprime{pair_i}.chance = max(dprime{pair_i}.chance, 1-dprime{pair_i}.chance);
                end
                
            end
            
            % s/f
            Y = BehaveData.success.indicatorPerTrial;
            X = pcaTrajres.projeff;
            dprime{pair_i + 1}.dprime = getDprime(X,Y);
            dprime{pair_i + 1}.labels = 'Success Failure';
            dprime{pair_i + 1}.trialsnum = length(Y);
            dprime{pair_i + 1}.chance = sum(Y==1)/length(Y);
            dprime{pair_i + 1}.chance = max(dprime{pair_i + 1}.chance, ...
                1-dprime{pair_i + 1}.chance);
            
            save(resfile, 'dprime');
        catch
            warning(['Problem with ' resfile ]);
        end
    end
end




end
