function [imagingData, BehaveData] = load_bda_tpa(BdaTpaList, generalProperty)

frameRateRatio = generalProperty.BehavioralSamplingRate/generalProperty.ImagingSamplingRate;

fileNumRoi = length(BdaTpaList);
[fileNamesRoi{1:fileNumRoi,1}] = deal(BdaTpaList.TPA);

for trialInd = 1:fileNumRoi
    disp(trialInd)
    usrData                    = load(fileNamesRoi{trialInd});
    if ~isfield(usrData, 'strROI')
        error([fileNamesRoi{trialInd} ' has no ROI']);
    end
    for m = 1:length(usrData.strROI)
        imagingDatatmp.roiNames{trialInd}{m}       = extractROIstr(usrData.strROI{m}.Name);
        % match new format to old format, the deltaF over F is saved in Data(:,2)
        % instead of procROI
        if ~isfield(usrData.strROI{m}, 'Data') && ~isprop(usrData.strROI{m}, 'Data')
            if ~isfield(usrData.strROI{m}, 'procROI')
                error([fileNamesRoi{trialInd} ' unfamiliar TPA file, cannot extract data';]);
            else
                imagingDatatmp.samples{trialInd}{m} = usrData.strROI{m}.procROI;
            end
        else
            imagingDatatmp.samples{trialInd}{m} = usrData.strROI{m}.Data(:,2);
            imagingDatatmp.loc{trialInd}{m} = usrData.strROI{m}.xyInd;
            
        end
    end
    
end
notthesameNrns = false;
Len = length(imagingDatatmp.roiNames{1});
for trial_i = 2:length(imagingDatatmp.roiNames)
    if Len ~= length(imagingDatatmp.roiNames{trial_i})
        notthesameNrns = true;
        break;
    end
end
if ~notthesameNrns
    reshapedROIs = reshape(cell2mat([imagingDatatmp.roiNames{:}]), length(imagingDatatmp.roiNames{1}), []);
    if any(any(diff(reshapedROIs.')))
           warning('ROI names are inconsistent through trials!');
           imagingData = [];BehaveData = [];
           return;
           notthesameNrns = true;
    end 
end
if notthesameNrns
    combinds = cell2mat(imagingDatatmp.roiNames{1});
    for trial_i = 2:length(imagingDatatmp.roiNames)
        combinds = intersect(combinds, cell2mat(imagingDatatmp.roiNames{trial_i}));
    end
    combinds = setdiff(combinds, 0);% avoid zero which is a place holder
    newImagingData = zeros(length(combinds), length(imagingDatatmp.samples{1}{1} ), length(imagingDatatmp.samples));
    ind2comb = findIndsLoc(combinds, cell2mat(imagingDatatmp.roiNames{1}));
    imagingData.loc(:, :, :) = imagingDatatmp.loc{1}(ind2comb);
    for trial_i = 1:length(imagingDatatmp.roiNames)
        ind2comb = findIndsLoc(combinds, cell2mat(imagingDatatmp.roiNames{trial_i}));
        for nr = 1:length(ind2comb)
        newImagingData(nr, :, trial_i) = imagingDatatmp.samples{trial_i}{ind2comb(nr)};
        end
    end
    
    imagingData.roiNames = repmat(combinds(:), 1, length(imagingDatatmp.roiNames));
    imagingData.samples = newImagingData;
    
    
else
   imagingData.roiNames = reshape(cell2mat([imagingDatatmp.roiNames{:}]), length(imagingDatatmp.roiNames{1}),[]);
   for trial_i = 1:length(imagingDatatmp.roiNames)
       for nr = 1:length(imagingDatatmp.roiNames{1})
           imagingData.samples(nr, :, trial_i) = imagingDatatmp.samples{trial_i}{(nr)};
           imagingData.loc{nr}(:, :,trial_i) = imagingDatatmp.loc{trial_i}{nr};
       end
    end
end
if ~isfield(BdaTpaList, 'BDA')
    BehaveData = [];
    return;
end
[fileNamesEvent{1:fileNumRoi,1}] = deal(BdaTpaList.BDA);


eventNameList = [];
allTrialEvents                = cell(fileNumRoi,1);
for trialInd = 1:fileNumRoi
    usrData                    = load(fileNamesEvent{trialInd});
    allTrialEvents{trialInd}   = usrData.strEvent;
    for event_i = 1:length(allTrialEvents{trialInd})
        if isempty(eventNameList) || ~any(strcmpi(eventNameList, allTrialEvents{trialInd}{event_i}.Name))
            eventNameList{end+1} = extractEventstr(allTrialEvents{trialInd}{event_i}.Name);
        end
    end
end
framNum = size(imagingData.samples,2);
for eventName_i = 1:length(eventNameList)
    if contains(eventNameList{eventName_i}, '*')
        eventNameList{eventName_i} = extractEventstr(eventNameList{eventName_i});
    end
    BehaveData.(eventNameList{eventName_i}).indicator = zeros(size(imagingData.samples,3), framNum);
    BehaveData.(eventNameList{eventName_i}).eventTimeStamps{size(imagingData.samples,3)} = []; 
end

for trial_i = 1:fileNumRoi
    for m = 1:length(allTrialEvents{trial_i})
        eventname = lower(allTrialEvents{trial_i}{m}.Name);
        eventname = extractEventstr(eventname);
        if length(allTrialEvents{trial_i}{m}.tInd) ==2
            timeInd     = allTrialEvents{trial_i}{m}.tInd;
        else
            timeInd     = allTrialEvents{trial_i}{m}.TimeInd;
        end
        if isempty(timeInd)
            continue;
        end
        %             frameRateRatio=size(allTrialEvents{trial_i}{end}.Data,1)/size(eventDataArray,1);
        %                 frameRateRatio=18
        timeInd     = round((timeInd-generalProperty.BehavioralDelay)./frameRateRatio); % transfers to time of the two photon
        timeInd     = max(1,min(framNum,timeInd));
        % assign to vector
        BehaveData.(eventname).indicator(trial_i, timeInd(1):timeInd(2)) = 1;
        BehaveData.(eventname).eventTimeStamps{trial_i} = timeInd;
    end
end
NAMES = fieldnames(BehaveData);
for name_i  =1:length(NAMES)
    [I,~] = find(BehaveData.(NAMES{name_i}).indicator);
    BehaveData.(NAMES{name_i}).indicatorPerTrial = zeros(fileNumRoi,1);
    BehaveData.(NAMES{name_i}).indicatorPerTrial(unique(I)) = 1;
end

if generalProperty.Neurons2keep ~= 0
    counter = 1;
    for nrind=1:length(generalProperty.Neurons2keep)
        curr_nrn2Keep = generalProperty.Neurons2keep(nrind);
        findResult = find(imagingData.roiNames(:, 1)-curr_nrn2Keep==0);
        if ~isempty(findResult)
            currnrnind(counter) = findResult;
            counter = counter + 1;
        end
    end
    
    imagingData.samples=imagingData.samples(currnrnind, :,:);
    imagingData.roiNames = imagingData.roiNames(currnrnind, 1);
end
% if generalProperty.Trials2keep == 0
%     return;
% end

for event_i = 1:length(eventNameList)
    switch eventNameList{event_i}
        case {'failure', 'success'}
            BehaveData.(eventNameList{event_i})=BehaveData.(eventNameList{event_i})(:);
        otherwise
            BehaveData.(eventNameList{event_i}).indicator=BehaveData.(eventNameList{event_i}).indicator(:, :);
    end
end

switch generalProperty.DetermineSucFailBy
    case 'suc'
        BehaveData.(generalProperty.failureLabel).indicatorPerTrial = zeros(size(BehaveData.(generalProperty.successLabel).indicatorPerTrial));
        BehaveData.(generalProperty.failureLabel).indicatorPerTrial(BehaveData.(generalProperty.successLabel).indicatorPerTrial == 0) = 1;
    case 'fail'
        BehaveData.(generalProperty.successLabel).indicatorPerTrial = zeros(size(BehaveData.(generalProperty.failureLabel).indicatorPerTrial));
        BehaveData.(generalProperty.successLabel).indicatorPerTrial(BehaveData.(generalProperty.failureLabel).indicatorPerTrial == 0) = 1;        
    case 'both'
        % do nothing
end
if isfield(BdaTpaList, 'traj')
    
for k = 1:length(BdaTpaList)
    C = xlsread(BdaTpaList(k).traj);
    
    if (generalProperty.do_Plot3DTraj)
        [~, ~, excelDataRaw] = xlsread(BdaTpaList(k).traj);
    
        [~, handFrontCol] = find(strcmp(excelDataRaw, 'LeftCenter'));   
        [~, handSideCol] = find(strcmp(excelDataRaw, 'RighCenter'));
    else
        handFrontCol = [2, 3];
        handSideCol = [5, 6];
    end
    
    BehaveData.traj.data(1,:,k) = C(:,handFrontCol(1));
    BehaveData.traj.data(2,:,k) = C(:,handFrontCol(2));
    BehaveData.traj.data(3,:,k) = C(:,handSideCol(1));
    BehaveData.traj.data(4,:,k) = C(:,handSideCol(2));
    
    if length(handFrontCol) > 2
%         likelhood
        BehaveData.traj.data(5,:,k) = C(:,handFrontCol(3));
        BehaveData.traj.data(6,:,k) = C(:,handSideCol(3));
    end
            
    
    clear C;
    
end
T = size(BehaveData.traj.data,2);
    filefacemap = fullfile(fileparts(fileparts(BdaTpaList(1).traj)),'facemap.mat');
    if exist(filefacemap, 'file')
        res = load(filefacemap);

        BehaveData.faceMapR = reshape(res.proc.motSVD{2}.', size(res.proc.motSVD{2},2), T, []);
        BehaveData.faceMapL = reshape(res.proc.motSVD{3}.', size(res.proc.motSVD{3},2), T, []);
    end
end


% if isfield(BdaTpaList, 'trk')
% for k = 1:length(BdaTpaList)
%     load(BdaTpaList(k).trk, '-mat');
%     BehaveData.trk.data(1,:,k) = pTrk(1,1,:);
%     BehaveData.trk.data(2,:,k) = pTrk(1,2,:);
%     BehaveData.trk.data(3,:,k) = pTrk(1,1,:);
%     BehaveData.trk.data(4,:,k) = pTrk(2,2,:);
%     clear pTrk;
% end
% end








