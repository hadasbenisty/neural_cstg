% organize all experiments into one tensor per animal
function collect_all_experiments(datapath, animals_names)
error('need to code this');
animals_db = get_animals_list(datapath, animals_names);


for animal_i = 1:length(animals_names)
    curranimal = fullfile(datapath, animals_names{animal_i});
    experiments = animals_db{animal_i}.folder;
    for fi = 1:length(experiments)
        datafile = fullfile(curranimal, experiments{fi}, 'data.mat');        
        if isfile(datafile)
            
            res = load(datafile);
            trials_inds{end+1} = res.trials_inds;
            data_v{end+1} = res.imagingData;
            if isfile(datafile_behave)
                donewbehavior(fi) = true;
                res1 = load(datafile_behave);
               
                be_v{end+1} = res1.BehaveData;
            else
%                 dobreak = true;
%                 break;
                be_v{end+1} = res.BehaveData;
            end
            training_lut{end+1} = res.training_stage;
            names{end+1} = res.imagingData.roiNames(:,1);
            training_labels_lut{end+1} = res.training_label;
            sessions_dates{end+1} = res.session_date;
        end
    end
    if dobreak
        continue;
    end
    disp(animals_names{animal_i})
    if animalsLabels(animal_i)
        animal_label = 'CNO';
    else
        animal_label = 'CTRL';
    end
    sflabels = [];train_stage = [];
    for k = 1:length(be_v)
        if ~isfield(be_v{k}, 'success')
            %be_v{k}.success.indicatorPerTrial = 1-be_v{k}.failure.indicatorPerTrial;
            currsf = 1-double(sum(be_v{k}.failure.indicator,2)>0);
        else
            currsf = double(sum(be_v{k}.success.indicator,2)>0);
        end
        sflabels = cat(1, sflabels, currsf);
        
        train_stage = cat(1, train_stage, ones(length(currsf), 1)*k);
    end
    [data_all, data_loc] = nrns_intersect(names, data_v);
    
    %% behavior
    if any(donewbehavior)
        names = [];
        for h = 1:length(be_v)
            names = cat(1, names, fieldnames(be_v{h}));
        end
        names = unique(names);
        names = setdiff(names, {'failure', 'success','tone','seq'});
        behaveData = zeros(length(names), behaviorT, size(data_all, 3));
        seq_data = [];
        for ni = 1:length(names)
            en = 0;
            for si = 1:length(be_v)
                st = en + 1;
                en = st + size(be_v{si}.tone.indicator, 1)-1;
                if isfield(be_v{si}, names{ni}) && size(be_v{si}.(names{ni}).indicator, 2) == 2400
                    behaveData(ni, :, st:en) = permute(be_v{si}.(names{ni}).indicator, [3, 2 1]);
                end
                if isfield(be_v{si}, 'seq')
                seq_data{si} = be_v{si}.seq;
                end
            end
        end
        behaviorLabels = names;
       
    else
        behaviorLabels = [];
        behaveData = [];
        seq_data = [];
        
    end
    %% save
    save(fullfile(datapath, animals_names{animal_i}, 'data7days_more_final.mat'), ...
        'seq_data','behaveData', 'behaviorLabels', ...
        'sessions_dates', ...
        'animal_label', 'data_loc', 'data_all', 'sflabels',...
        'train_stage', 'training_lut', 'training_labels_lut');
    
end
