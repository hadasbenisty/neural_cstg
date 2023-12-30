clear all;
clc;
close all;

% params to set by user:
compute_rname = 'shira'; % or 'hadas' or zohar
animals_names = {'4458'}; %{'4575' '4756',  '4754',  '4458' '4882' '4880'   };%'4940'

% code:
addpath(genpath('utils'));
addpath(genpath('manage_data'));
addpath(genpath('svm'));
addpath(genpath('temporalTraj'));

switch compute_rname
    case 'hadas'
        datapath = '../../';
    case 'zohar'
        datapath = 'F:\Dropbox (Technion Dropbox)\Zohar\Analysis\';
    case 'shira'
        datapath = '..\data\';
    otherwise
        error('Unfamiliar computer name');
end

resultspath = '../analysisres/';
figsfolder = '../figs2/';
excelpath = '../xls2/';
mkNewFolder(resultspath);
mkNewFolder(figsfolder);
mkNewFolder(excelpath);
clc

collectData(datapath, animals_names);% save data from BDA TPA to mat
% do not use collect_all_experiments(datapath, animals_names);% save data from all days in a single mat file

% pca trajectories
pcaTrajectories(datapath, resultspath, animals_names);
plotPcaTrajectories(datapath, resultspath, animals_names, figsfolder)
dPrime_analysis(datapath, resultspath, animals_names);
dPrime_analysis_q_vs_all(datapath, resultspath, animals_names);

plot_flavors_dprime(datapath, resultspath, animals_names, figsfolder);
plot_flavors_dprime_q_vs_all(datapath, resultspath, animals_names, figsfolder);

% svm
flavors_svm(datapath, resultspath, animals_names)
flavors_svm_q_vs_all(datapath, resultspath, animals_names)

plot_flavors_svm(datapath, resultspath, animals_names, figsfolder);
plot_flavors_svm_q_vs_all(datapath, resultspath, animals_names, figsfolder);
% flavors_svm_by_cell_trace(datapath, resultspath, animals_names)

flavors_svm_by_cell(datapath, resultspath, animals_names)
flavors_svm_by_cell_q_vs_all(datapath, resultspath, animals_names)

plot_indicative_fraction_flavor(datapath, resultspath, animals_names, figsfolder, excelpath)

plot_flavors_svm_by_cell(datapath, resultspath, animals_names, figsfolder, excelpath)
plot_flavors_svm_by_cell_summary(datapath, resultspath, animals_names, figsfolder, excelpath)
plot_flavors_svm_by_cell_summary_q_vs_all(datapath, resultspath, animals_names, figsfolder, excelpath)
plot_flavors_and_sf_svm_by_cell_summary(datapath, resultspath, animals_names, figsfolder, excelpath)
