function effectiveDim = getEffectiveDim(vals, th)

% inds=find(vals(1:end-1)./vals(2:end)-1<th);
inds=find(cumsum(vals.^2)/sum(vals.^2)>th);
if isempty(inds)
    effectiveDim = length( vals);
else
    effectiveDim = inds(1);
end
