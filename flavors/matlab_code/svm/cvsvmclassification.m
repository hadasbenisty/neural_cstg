function [bestc, bestg] = cvsvmclassification(X, Y, C, G, foldsNum)
cv = zeros(length(C), length(G));
classes = unique(Y);
wstr=' ';
for w_i=1:length(classes)
    w = sum(Y==classes(w_i))/length(Y);
    wstr=[wstr ' -w' num2str(classes(w_i)) ' ' num2str(w)];
end
if isempty(G)
    for Cind = 1:length(C)
        cv(Cind) = svmtrain(Y, sparse(X), [' -t 0 -v ' num2str(foldsNum) ' -c ', num2str(C(Cind))  ' ' wstr ' -q'] );
    end
else
    for Gind = 1:length(G)
        for Cind = 1:length(C)
            cv(Cind, Gind) = svmtrain(Y, sparse(X), [' -t 2 -v ' num2str(foldsNum) ' -c ', num2str(C(Cind)) ' -g ' num2str(G(Gind)) wstr ' -q'] );
        end
    end
end
if isempty(G)
    if all(cv == max(cv(:)))
        bestc=1;bestg=1;
    else
        I = find(cv == max(cv(:)));
        bestc = C(I(1));
        bestg = 1;
    end
else
    if all(cv == max(cv(:)))
        bestc=1;bestg=1;
    else
        [I,J] = find(cv == max(cv(:)));
        bestc = C(I(1));
        bestg = G(J(1));
    end
end