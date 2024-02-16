function [chanceLevel, estimated_level] = naive_expert_svm_CC ...
    (data_all, train_stage, training_labels_lut)
% ---- Descripition ----
% The function recieves the total data (the activity of each neuron at each
% time measurement at each trial) and tries to build a metric to classify
% the expertee of the animal during the learning. It does so by training
% the data for a SVM on the first and last session which we assume are
% naive and expert. Then this machine is tested on all trials and we
% recieve a tensor called predictions. Then we calculate the mean value
% along the third dimention (the dimention the size of folds). Which gives
% us the matrix expertEstimation. Using the matrix we calculate the
% estimated_level matrix which is defined as the sum of all the estimations
% we had in that session in that point of the time divided by all the
% estimation we had for that session. (we normalize the rows).
% ---- Inputs ----
% data_all: a neurons x time_measurements x number_of_trials (total). It is
% holding the total data of the experiment, not divived into differnet
% sessions.
% train_stage: a vector the size number_of_trials x 1 that holds at each
% element the number of session this current trial relates to.
% training_labels_lut: a 1 x number_of_different_sessions cell array that
% holds the names for each train session.
% ---- Outputs ----
% chanceLevel - a double, hold the ratio between the number of expert
% trials to the total amount of naive trials and expert trials.
% estimated_level - a number_of_sessions x number_of_windows, holds the
% estimated level between beginner and expert (beginner - 0 and expert  -
% 1). 
% ---- Importatnt Local Parameters ----
% params: a struct that holds there properties:
%%%%
% duration - a double, the duration of each trial.
% slidingWinLen - a double, the length of each window
% slidingWinHop - a double, the jump between each window, meaning they can
% overlap
% foldsnum -
% islin - a boolean, 
% tone - a double, probably the time we started before the actual 0.
% naiveLabel - a string, the name of the first session.
%%%%
% winstSec - a vector the size of 1 x floor(params.duration / params.hop)
% that holds the starting time for each window.
% winendSec - a vector the size of 1 x floor(params.duration / params.hop)
% that holds the ending time for each window.
% t -  a vector, the size of 1 x time_measurements, holds the time of each
% time measurment. 
% predictions - a tensor the size of number_of_trials x number_of_windows x
% foldsNum, holds for each element a value of naiveIndex or expertIndex,
% according to the svm prediction.
% labs - a vector the size of 1 x number_of_session, each element is a
% different session number.
% expertEstimation - a matrix the size of number_of_trials x
% number_of_windows, holds the estimation we have made for each window for
% each trial on whether the animal has preformed well then.


params.duration = 12;

params.slidingWinLen = params.duration; params.slidingWinHop = params.duration;
params.foldsnum = 10; params.islin = 1;params.tone=4;
params.naiveLabel = 'train1';




expertindex = max(train_stage);
naiveindex = 1;

% [winstSec, winendSec] = getFixedWinsFine(params.duration, params.slidingWinLen, params.slidingWinHop);
winstSec = 0;
winendSec = 1;
t = 0.5;
chanceLevel = sum(train_stage==expertindex)/ ...
    sum(train_stage==naiveindex|train_stage==expertindex);
chanceLevel=max(chanceLevel,1-chanceLevel);

predictions = train_test_naive_expert_svm_sliding(data_all, t, winstSec, winendSec, 1, ...
    train_stage, naiveindex,  expertindex);
labs = unique(train_stage);
expertEstimation = mean(predictions==expertindex,3);
tmid = mean([winstSec;winendSec]) - params.tone;
estimated_level = nan(length(labs), length(tmid));
for ti=1:size(predictions,2)
    for ci = 1:length(labs)
        estimated_level(ci,ti) = sum(expertEstimation( ...
            train_stage==labs(ci),ti))/sum(train_stage==labs(ci));
    end

end

end

