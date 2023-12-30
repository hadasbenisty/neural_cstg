function pcaTrajectories(datapath, resultspath, animals_names)
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
        if isfile(resfile)
            continue;
        end
        load(datafile, 'imagingData', 'BehaveData');
        for k=1:size(imagingData.samples,1)
            alldataNT(:, k) = reshape(imagingData.samples(k,:,:), ...
                size(imagingData.samples,3)*size(imagingData.samples,2),1);
        end
        pca_thEffDim = 0.95;
        
        
        [pcaTrajres.kernel, pcaTrajres.mu, pcaTrajres.eigs] = mypca(alldataNT);
        pcaTrajres.effectiveDim = getEffectiveDim(pcaTrajres.eigs, ...
            pca_thEffDim);
        [recon_m, projeff] = linrecon(alldataNT, pcaTrajres.mu, ...
            pcaTrajres.kernel, 1:pcaTrajres.effectiveDim);
        if pcaTrajres.effectiveDim < 3
            [~, projeff] = linrecon(alldataNT, pcaTrajres.mu, pcaTrajres.kernel, 1:3);
        end
        for l=1:size(recon_m,2)
            pcaTrajres.recon(l,:,:) = reshape(recon_m(:,l),size(imagingData.samples,2), ...
                size(imagingData.samples,3));
        end
        for l=1:size(projeff,2)
            pcaTrajres.projeff(l,:,:) = reshape(projeff(:,l), size(imagingData.samples,2), ...
                size(imagingData.samples,3));
            
        end
        save(resfile, 'pcaTrajres');
        clear alldataNT;
        clear pcaTrajres;
    end
end

