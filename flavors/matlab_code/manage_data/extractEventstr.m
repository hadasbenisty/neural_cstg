function eventName = extractEventstr(fullstr)
inds = union(strfind(fullstr, ':'),strfind(fullstr, ' '));
leagalInds = setdiff(1:length(fullstr), inds);
eventName=lower(fullstr(leagalInds));
eventName(eventName=='-') = '_';
eventName(eventName=='+') = 'p';
i = findstr(eventName, '*');
if ~isempty(i)    
eventName = [eventName(1:i-1) '_ignore'];
end




