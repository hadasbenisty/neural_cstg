function [predictions, models, accSVM] = train_test_naive_expert_svm_sliding(X, t, winstSec, winendSec, tonorm, Y, naiveLabel,  expertLabel)
accSVM.raw.mean=[];accRandSVM.raw.mean=[]; models=[];
foldsnum=10;islin=1;
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
    rawX_naive_expert = Xnorm(Y==naiveLabel|Y==expertLabel,:);
    YnaiveExpert = Y(Y==naiveLabel|Y==expertLabel);
    YnaiveExpert1=YnaiveExpert;
    YnaiveExpert1(YnaiveExpert==1) = 0;
    YnaiveExpert1(YnaiveExpert==7) = 1;



    if isempty(accSVM.raw) ||  length(accSVM.raw.mean) < win_i
        [ ACC, ACCrand, ~, ~, ~, ~, ~, ~, ~, SVMModel] = ...
            svmClassifyAndRand(rawX_naive_expert, YnaiveExpert, YnaiveExpert, foldsnum, '', islin, false);
        [accSVM.raw, accRandSVM.raw] = setStats(ACC, ACCrand, win_i, accSVM.raw, accRandSVM.raw);
    end
    models{win_i} = SVMModel;

    for fi = 1:length(models{win_i})
        predictions(:,win_i, fi) = svmpredict(expertLabel*double(Y==expertLabel), sparse(Xnorm), models{win_i}(fi));
    end
end
end


function [accSVM, accRandSVM] = setStats(ACC, ACCrand, win_i, accSVM, accRandSVM)
accSVM.mean(win_i) = ACC.mean;
accSVM.std(win_i) = ACC.std;
accSVM.accv(:, win_i) = ACC.acc_v;
% raw rand
accRandSVM.mean(win_i) = ACCrand.mean;
accRandSVM.std(win_i) = ACCrand.std;
accRandSVM.accv(:, win_i) = ACCrand.acc_v;
end
