function dPrime_analysis_q_vs_all(datapath, resultspath, animals_names)

animals_db = get_animals_list(datapath, animals_names);
addpath(genpath('pca'));
flavors = {'sucroses', 'regulars', 'grains','fakes'};
qlabel = 'quinines';

for animal_i = 1:length(animals_names)
    disp(animals_names{animal_i});
    datesList = animals_db{animal_i}.folder;
    
    
    for ei = 1:length(datesList)
        
            disp(datesList{ei});
            currfolder = fullfile(datapath, animals_names{animal_i}, datesList{ei});
            datafile = fullfile(currfolder, 'data.mat');
            resfilepca = fullfile(resultspath, ['pca_trajectories_' animals_names{animal_i} '_' datesList{ei} '.mat']);
            
            if ~isfile(datafile) || ~isfile(resfilepca)
                continue;
            end
            resfile = fullfile(resultspath, ['dprime_trajectories_' animals_names{animal_i} '_' datesList{ei} '_q_vs_all.mat']);
            if isfile(resfile)
                continue;
            end
            load(resfilepca, 'pcaTrajres');
            load(datafile, 'BehaveData');
            if ~isfield(BehaveData, qlabel)
                continue;
            end
            Y = BehaveData.(qlabel).indicatorPerTrial;
            Y(Y==1) = 2;
            
            for f_i = 1:length(flavors)
                if isfield(BehaveData, flavors{f_i})
                    Y(BehaveData.(flavors{f_i}).indicatorPerTrial==1) = 1;
                end
            end
            X = pcaTrajres.projeff(:, :, Y > 0);
            Y = Y(Y > 0);
            Y = Y - 1;
            if ~isempty(X) && ~isempty(Y)
                dprime.dprime = getDprime(X,Y);
                dprime.labels = [qlabel ' all'];
                dprime.trialsnum = length(Y);
                dprime.chance = sum(Y==1)/length(Y);
                dprime.chance = max(dprime.chance, 1-dprime.chance);
            end
            
        
        save(resfile, 'dprime');
        
    end
end




end
