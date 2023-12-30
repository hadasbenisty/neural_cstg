function [count, I] = getIndicativeNrnsMean(isindicative, method, binsnum, time2st, time2end)

switch method
    case 'consecutive'
        isindicativediff = isindicative(:,2:end)+isindicative(:,1:end-1);
        indicatorpernrn = sum(isindicativediff(:,time2st:time2end-1)>binsnum-1,2);
        count = sum(indicatorpernrn>0)/length(indicatorpernrn)*100;
        I = indicatorpernrn>0;
    case 'any'
    isindicPerDay = sum(isindicative(:,time2st:time2end),2);
    count= sum(isindicPerDay>=binsnum)/length(isindicPerDay)*100;
I = isindicPerDay>=binsnum;
    otherwise
        error('Unfamilar method for evaluating averaged indicative neurons Choose ''consecutive'' or ''any''');
end
