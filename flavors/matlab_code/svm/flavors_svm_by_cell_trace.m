function flavors_svm_by_cell_trace(datapath, resultspath, animals_names)

addpath(genpath('svm'));
animals_db = get_animals_list(datapath, animals_names);
fsample = 30;
params.slidingWinLen = 1;
params.slidingWinHop = 0.5;
params.foldsnum = 10;
params.tonorm = 1;
flavors = { 'quinines', 'sucroses', 'regulars', 'grains','fakes'};
pairs = nchoosek(1:length(flavors), 2);
winstSec = [0 4 7];
winendSec = [4 7 20];
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
        resfile = fullfile(resultspath, ['svm_by_cell_' animals_names{animal_i} '_' datesList{ei} 'trace.mat']);
        if isfile(resfile)
            continue;
        end
        load(datafile, 'imagingData', 'BehaveData');
        if ~isfield(BehaveData, 'success')
            warning(['No success for ' animals_names{animal_i} ' ' datesList{ei}]);
            continue;
        end
        t = (0:size(imagingData.samples, 2)-1)/fsample;
        
%         [winstSec, winendSec] = getFixedWinsFine(round(t(end)), params.slidingWinLen, params.slidingWinHop);
        acc = cell(size(pairs, 1)+1, 1);
        for pair_i = 1:size(pairs, 1)
            [X, Y] = get_data(imagingData.samples, BehaveData, flavors{pairs(pair_i, 1)}, ...
                flavors{pairs(pair_i, 2)});
            if ~isempty(X) && ~isempty(Y)
                acc{pair_i} = svm_by_cell(X, Y, winstSec, winendSec, t, params.foldsnum, params.tonorm);
                acc{pair_i}.labels = [flavors{pairs(pair_i, 1)} ' ' flavors{pairs(pair_i, 2)}];
                acc{pair_i}.trialsnum = length(Y);
                acc{pair_i}.chance = sum(Y==1)/length(Y);
                acc{pair_i}.chance = max(acc{pair_i}.chance, 1-acc{pair_i}.chance);
            end
            
        end
        
        % s/f
        Y = BehaveData.success.indicatorPerTrial;
        X = imagingData.samples;
        acc{pair_i + 1} = svm_by_cell(X, Y, winstSec, winendSec, t, params.foldsnum, params.tonorm);
        acc{pair_i + 1}.labels = 'success failure';
        acc{pair_i + 1}.chance = sum(Y==1)/length(Y);
        acc{pair_i + 1}.chance = max(acc{pair_i + 1}.chance, 1-acc{pair_i + 1}.chance);
        acc{pair_i + 1}.trialsnum = length(Y);
        
        save(resfile, 'acc');
    end
end

end
function acc = svm_by_cell(X, Y, winstSec, winendSec, t, foldsnum, tonorm)

% predict all trials using the models trained for naive/expert
for win_i = 1:length(winstSec)
    
    Xwin = X(:,t >= winstSec(win_i) & t <= winendSec(win_i),:);
    
    
    
    
    
    for nrni = 1:size(Xwin, 1)
        rawX = squeeze(Xwin(nrni, :, :))';
        if tonorm
            Xnorm = (rawX - min(rawX(:)))/(max(rawX(:))-min(rawX(:)));
        else
            Xnorm=X;
        end
        if length(Xnorm) == numel(Xnorm)
            Xnorm=Xnorm(:);
        end
       CVSVMModel = fitcsvm(rawX,Y,'Standardize',true,'CrossVal','on');
genError = kfoldLoss(CVSVMModel);
Mdl = fitckernel(rawX,Y)
SVMModel = fitcsvm(rawX,Y,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');

        ACC = svmClassifyAndRand(Xnorm, Y, Y, foldsnum, '', true, false);
        acc.mean(nrni, win_i) = ACC.mean;
        acc.std(nrni, win_i) = ACC.std;
        acc.accv(nrni, :, win_i) = ACC.acc_v;
    end
    
end
end