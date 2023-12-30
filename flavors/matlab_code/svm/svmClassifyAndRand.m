function [ acc, accrand, scoresall, scoresallrand, confMat, confMatrand, ...
    accclasses, accclassesrand, Yhat, SVMModel] = svmClassifyAndRand(X, Y, Yrand, foldsNum, filename, islin, tonorm)
if ~exist('islin', 'var')
    islin = false;
end
scoresall=[];scoresallrand=[];
if exist(filename, 'file')
    load(filename)
    return;
end
X=X.';
if size(X,2)==1
        X = X.';
    end
if size(X,2) ~= length(Y)
    
    error('Dims are inconsistent');
end
if foldsNum >= length(Y)
    INDICES = 1:length(Y);
else
    INDICES = crossvalind('Kfold',length(Y),foldsNum);
end
if tonorm
Xnorm = (X - min(X(:)))/(max(X(:))-min(X(:)));
else 
    Xnorm=X;
end
% pb = CmdLineProgressBar('SVM CV');
log2c = -6:10;log2g = -6:4;
classes = unique(Y);
wstr=' ';
for w_i=1:length(classes)
    w = sum(Y==classes(w_i))/length(Y);
    wstr=[wstr ' -w' num2str(classes(w_i)) ' ' num2str(w)];
end
for fold_i = 1:length(unique(INDICES))
%     pb.print(fold_i,foldsNum);
    testinds = find(INDICES == fold_i);
    cvinds = setdiff(1:length(Y), testinds);
    if islin
        bestc(fold_i) = cvsvmclassification(Xnorm(:,cvinds).', Y(cvinds), 2.^log2c, [], foldsNum);        
        SVMModel(fold_i) = svmtrain(Y(cvinds), Xnorm(:,cvinds)', ['-t 0 -q  -c ', num2str(bestc(fold_i)) ' ' wstr] );
    else
[bestc(fold_i), bestg(fold_i)] = cvsvmclassification(Xnorm(:,cvinds).', Y(cvinds), 2.^log2c, 2.^log2g, foldsNum);
SVMModel(fold_i) = svmtrain(Y(cvinds), Xnorm(:,cvinds)', ['-t 2 -q  -c ', num2str(bestc(fold_i)) ' -g ' num2str(bestg(fold_i)) wstr] );
    
    end
    [predictions, ~, score] = svmpredict(Y(testinds), sparse(Xnorm(:,testinds))', SVMModel(fold_i));
    acc_v(fold_i) = sum(predictions==Y(testinds))/length(testinds);
    Yhat(testinds) = predictions;
%     if nargout > 2
%     scoresall(testinds,:)=score;
%     end
    [predictions, ~, score] = svmpredict(Yrand(testinds), sparse(Xnorm(:,testinds))', SVMModel(fold_i));
    acc_vrand(fold_i) = sum(predictions==Yrand(testinds))/length(testinds);
    Yhatrand(testinds) = predictions;
%     if nargout > 2
%     scoresallrand(testinds,:)=score;
%     end
    %     if length(unique(Y(testinds))) < 2
    %         auc_v(fold_i)=nan;
    %     else
    %         [fpRate,hitRate,T,auc_v(fold_i)] = perfcurve(Y(testinds),score,SVMModel.Label(1));
    %     end
    
    a = getAccByClass(Yrand(testinds), Yhatrand(testinds));
    cl = unique(Yrand(testinds));
    for k=1:length(classes)
        if  any((cl== classes(k)))
            accclassesrand(fold_i,k) = a(cl== classes(k));
        end
    end
    
    a = getAccByClass(Y(testinds), Yhat(testinds));
    cl = unique(Y(testinds));
    for k=1:length(classes)
        if any((cl== classes(k)))
            accclasses(fold_i,k) = a(cl== classes(k));
        end
    end
end
acc.acc_v = acc_v;
% acc.mean = sum(Y(:)==Yhat(:))/length(Y);
acc.mean = mean(acc_v(~isnan(acc_v)));
acc.std  = std(acc_v(~isnan(acc_v)));
accrand.mean = mean(acc_vrand(~isnan(acc_vrand)));
accrand.std  = std(acc_vrand(~isnan(acc_vrand)));
accrand.acc_v = acc_vrand;

confMat=confusionmat(Y,Yhat);
confMatrand=confusionmat(Yrand,Yhatrand);
if ~isempty(filename)
save(filename,'SVMModel','bestc','bestg','acc', 'accrand', 'scoresall', 'scoresallrand', 'confMat', 'confMatrand', 'accclasses', 'accclassesrand');
end
end

function accclasses = getAccByClass(Y, Yhat)
classes = unique(Y);
for c_i = 1:length(classes)
    inds = find(Y== classes(c_i));
    accclasses(c_i) = sum(Yhat(inds) == classes(c_i))/length(inds);
end
end

