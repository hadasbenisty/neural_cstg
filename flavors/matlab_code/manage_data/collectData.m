function collectData(datapath, animals_names)

%% organize all experiments into a 3D matrix (per animal)
animals_db = get_animals_list(datapath, animals_names);
generalProperty.BehavioralSamplingRate = 200;
generalProperty.ImagingSamplingRate = 30;
generalProperty.BehavioralDelay = 20;
generalProperty.Neurons2keep = 0;
generalProperty.DetermineSucFailBy = 'both';
for animal_i = 1:length(animals_names)
    disp(animals_names{animal_i});
    datesList = animals_db{animal_i}.folder;
    
   
    for ei = 1:length(datesList)
        disp(datesList{ei});
        currfolder = fullfile(datapath, animals_names{animal_i}, datesList{ei});
        if ~isfolder(currfolder)
            disp([currfolder ' is missing']);
            continue;
        end
        
        if ~isfile(fullfile(currfolder, 'data.mat'))
            bdalist = dir(fullfile(currfolder, 'BDA_TSeries_*.mat'));
            if isempty(bdalist)
                disp(['no bda for ' currfolder]);
                continue;
            end
            tpalist = dir(fullfile(currfolder, 'TPA*.mat'));
            if isempty(tpalist)
                disp(['no tpa for ' currfolder']);
                continue;
            end
            trialindBDA=[];
            for l = 1:length(bdalist)
                si = strfind(bdalist(l).name, '_Cycle');
                trialindBDA(l) = str2num(bdalist(l).name(si-3:si-1));
            end
            trialindTPA = [];
            for l = 1:length(tpalist)
                si = strfind(tpalist(l).name, '_Cycle');
                trialindTPA(l) = str2num(tpalist(l).name(si-3:si-1));
            end
            trials_inds = intersect(trialindTPA, trialindBDA);
            BdaTpaList=[];
            for trial = 1:length(trials_inds)
                curr_trial = trials_inds(trial);
                BdaTpaList(trial).BDA = fullfile(bdalist(curr_trial==trialindBDA).folder, bdalist(curr_trial==trialindBDA).name);
                BdaTpaList(trial).TPA = fullfile(tpalist(curr_trial==trialindTPA).folder, tpalist(curr_trial==trialindTPA).name);
            end
%             try
            [imagingData, BehaveData] = load_bda_tpa(BdaTpaList, generalProperty);
            if isempty(imagingData) || isempty(BehaveData)
                continue;
            end
            save(fullfile(currfolder, 'data.mat'),'imagingData', 'BehaveData');
%             catch ME
%                 warning(ME.message);
%                 warning(['Problem with ' currfolder]);
%             end
        end 
        
    end
end