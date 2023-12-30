function indLoc = findIndsLoc(items2loc, List)

for k = 1:length(items2loc)
    indLoc(k) = find(List-items2loc(k) == 0);
end