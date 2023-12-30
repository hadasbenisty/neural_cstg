function flavors_svm_q_vs_all(datapath, resultspath, animals_names)

addpath(genpath('svm'));
animals_db = get_animals_list(datapath, animals_names);
fsample = 30;
params.slidingWinLen = 1;
params.slidingWinHop = 0.5;
params.foldsnum = 10;
params.tonorm = 1;
flavors = {'sucroses', 'regulars', 'grains','fakes'};
qlabel = 'quinines';

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
        resfile = fullfile(resultspath, ['svm_' animals_names{animal_i} '_' datesList{ei} '_q_vs_all.mat']);
        if isfile(resfile)
            continue;
        end
        load(datafile, 'imagingData', 'BehaveData');
        if ~isfield(BehaveData, qlabel)
            continue;
        end
        t = (0:size(imagingData.samples, 2)-1)/fsample;
        Y = BehaveData.(qlabel).indicatorPerTrial;
        Y(Y==1) = 2;
        
        for f_i = 1:length(flavors)
            if isfield(BehaveData, flavors{f_i})
                Y(BehaveData.(flavors{f_i}).indicatorPerTrial==1) = 1;
            end
        end
        X = imagingData.samples(:, :, Y > 0);
        Y = Y(Y > 0);
        Y = Y - 1;
        [winstSec, winendSec] = getFixedWinsFine(round(t(end)), params.slidingWinLen, params.slidingWinHop);
        
        
        
        if ~isempty(X) && ~isempty(Y)
            acc = sliding_svm(X, Y, winstSec, winendSec, t, params.foldsnum, params.tonorm);
            acc.labels = [qlabel ' all'];
            acc.trialsnum = length(Y);
            acc.chance = sum(Y==1)/length(Y);
            acc.chance = max(acc.chance, 1-acc.chance);
        end
        save(resfile, 'acc');
    end
    
    
    
    
end


end
function acc = sliding_svm(X, Y, winstSec, winendSec, t, foldsnum, tonorm)

% predict all trials using the models trained for naive/expert
for win_i = 1:length(winstSec)
    
    Xwin = X(:,t >= winstSec(win_i) & t <= winendSec(win_i),:);
    rawX=squeeze(mean(Xwin,2))';
    if tonorm
        Xnorm = (rawX - min(rawX(:)))/(max(rawX(:))-min(rawX(:)));
    else
        Xnorm=X;
    end
    if length(Xnorm) == numel(Xnorm)
        Xnorm=Xnorm(:);
    end
    
    
    
    
    ACC = svmClassifyAndRand(Xnorm, Y, Y, foldsnum, '', true, false);
    acc.mean(win_i) = ACC.mean;
    acc.std(win_i) = ACC.std;
    acc.accv(:, win_i) = ACC.acc_v;
    
    
end
end