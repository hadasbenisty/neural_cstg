function plotPcaTrajectories(datapath, resultspath, animals_names, figsfolder)
animals_db = get_animals_list(datapath, animals_names);
addpath(genpath('pca'));
for animal_i = 1:length(animals_names)
    disp(animals_names{animal_i});
    datesList = animals_db{animal_i}.folder;
    
    
    for ei = 1:length(datesList)
        disp(datesList{ei});
        currfolder = fullfile(datapath, animals_names{animal_i}, datesList{ei});
        datafile = fullfile(currfolder, 'data.mat');
        if ~isfile(datafile)
            continue;
        end
        resfile = fullfile(resultspath, ['pca_trajectories_' animals_names{animal_i} '_' datesList{ei} '.mat']);
        if ~isfile(resfile)
            continue;
        end
        load(datafile, 'BehaveData');
        load(resfile);
        
        
        if ~isfield(BehaveData, 'success') && ~isfield(BehaveData, 'failure')
                continue;
            end
        figure;
        
        if isfield(BehaveData, 'success')
            subplot(2,2,1);
            labels = BehaveData.success.indicatorPerTrial+1;
            plot_trials_traj(pcaTrajres.projeff(1:3, 20:end, labels>0), labels(labels>0), 'rb', {'Failure'  'Success'});
        else
            subplot(2,2,1);
            labels = (1-BehaveData.failure.indicatorPerTrial)+1;
            plot_trials_traj(pcaTrajres.projeff(1:3, 20:end, labels>0), labels(labels>0), 'rb', {'Failure'  'Success'});
        end
        flavors = {'sucroses' 'quinines' 'grains' 'regulars' 'fakes'};
        labels1 = double(labels == 1);
        for kk = 1:length(flavors)
            includeflavor(kk) = isfield(BehaveData, flavors{kk});
            if includeflavor(kk)
                labels1(BehaveData.(flavors{kk}).indicatorPerTrial == 1) = 1+kk;
                
            end
        end
        if any(includeflavor)
            subplot(2,2,2);
            plot_trials_traj(pcaTrajres.projeff(1:3, 20:end, labels>0), ...
                labels1(labels1>0),  'rgmcky', cat(2, {'Failure'},  flavors{includeflavor}));
        end
        
        flavors = {'sucrose' 'quinine' 'grain' 'regular' 'fake'};
        labels2 = zeros(size(labels1));
        for kk = 1:length(flavors)
            includeflavor(kk) = isfield(BehaveData, flavors{kk});
            if includeflavor(kk)
                labels2(BehaveData.(flavors{kk}).indicatorPerTrial == 1) = kk;
                
            end
        end
        if any(includeflavor)
            subplot(2,2,3);
            plot_trials_traj(pcaTrajres.projeff(1:3, 20:end, labels2>0), ...
                labels2(labels2>0),  'gmcky', flavors(includeflavor));
        end
        
        mysave(gcf, fullfile(figsfolder, 'per_animal_stats', ...
            [animals_names{animal_i}, '_' datesList{ei} '_pca_traj']));
        close all;
    end
    %        figure;
    %        X1 = squeeze(mean(pcaTrajres.projeff(:, 400:end, :), 2));
    %        X2 = squeeze(mean(pcaTrajres.projeff(:, 20:200, :), 2));
    %
    %        labels = BehaveData.success.indicatorPerTrial+1;
    %        clrs = 'rb';
    %        subplot(2,3,1);
    %        X3 = X1(:, labels>0);
    %        labels = labels(labels > 0);
    %        plot_trials_dots(X3, labels, clrs, {'Failure'  'Success'});
    %        subplot(2,3,4);
    %        X3 = X2(:, labels>0);
    %        plot_trials_dots(X3, labels, clrs, {'Failure'  'Success'});
    %
    %
    %        labels = BehaveData.sucrose.indicatorPerTrial + ...
    %            BehaveData.grain.indicatorPerTrial*2;
    %         X3 = X1(:, labels>0);
    %        labels = labels(labels > 0);
    %        clrs = 'mg';
    %        subplot(2,3,2);
    %        plot_trials_dots(X3, labels, clrs, {'Sucrose'  'Grain'});
    %         X3 = X2(:, labels>0);
    %       subplot(2,3,5);
    %        plot_trials_dots(X3, labels, clrs, {'Sucrose'  'Grain'});
    %
    %
    %         labels = BehaveData.failure.indicatorPerTrial + ...
    %            BehaveData.grains.indicatorPerTrial*2+BehaveData.sucroses.indicatorPerTrial*3;
    %        clrs = 'rgm';
    %        X3 = X1(:, labels>0);
    %        labels = labels(labels > 0);
    %        subplot(2,3,3);
    %        plot_trials_dots(X3, labels, clrs, {'Failure'  'Grain' 'Sucrose'});
    %        X3 = X2(:, labels>0);
    %        subplot(2,3,6);
    %        plot_trials_dots(X3, labels, clrs, {'Failure'  'Grain' 'Sucrose'});
    %
    
    clear alldataNT;
    clear pcaTrajres;
end
end

