import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import scipy.io as spio
from scipy import stats
import torch.utils.data as data_utils
import torch
from flavors.utils import norm_minmax


class DateProcessor:
    def __init__(self, params, matfile_path, relevent_idx, date_info):

        self.data = spio.loadmat(matfile_path)
        self.date_info = date_info  # Store additional information about the date
        if date_info.iloc[0]['codition'] == 'control' and date_info.iloc[0]['to_include'] == 2 \
                and date_info.iloc[0]['type1'] == 'ongoing':
            self.use_flag = True
        else:
            self.use_flag = False
        self.relevent_idx = relevent_idx  # only indices that intersects for all dates
        # Initialization
        self.explan_feat = None  # after multiply by contextual gates will be input to the prediction model
        self.context_feat = None  # inputs to the hyper-network that creates mu vector
        self.output_label = None  # labels
        self.trials = None

        self.outcome_options = params.outcome_options
        self.start_time = params.start_time  # -4 sec
        self.end_time = params.end_time   # 8 sec
        self.drop_time = params.drop_time  # drop the first sec
        self.sample_per_sec = params.sample_per_sec  # samples/sec
        self.window_size_avg = params.window_size_avg  # 1 sec
        self.overlap_avg = params.overlap_avg  # 0.5 sec

    def get_output_labels(self, outcome_keys):

        outcomes = []
        for key in outcome_keys:
            outcome_label = self.data['BehaveData'][key][0][0]['indicatorPerTrial'][0][0]
            outcomes.append(outcome_label)

        # Stack the arrays horizontally to create a 2D array
        stacked_outcomes = np.column_stack(outcomes)
        # Create a new array based on the 'And' condition, only trials that satisfy all keys are indicate with '1'
        output_labels = np.all(stacked_outcomes, axis=1).astype(int)

        return output_labels  # vector size number of trails

    def get_explan_feat(self, relative_idx_neu, window_size, overlap, total_samples):

        explan_feat = np.empty((0, len(self.relevent_idx)))

        i = 1
        start_time = self.drop_time * self.sample_per_sec - 1  # drop the first sec
        for tind in range(start_time, total_samples - window_size + 1, overlap):
            # Extract the window of data for the moving average
            win_x = np.squeeze(self.data['imagingData']['samples'][0][0][relative_idx_neu, tind:tind+window_size, :])
            # Calculate the average along the time axis (axis=1)
            x = np.mean(win_x, axis=1)
            # Append the averaged data to explan_feat
            explan_feat = np.vstack((explan_feat, x.T))
            i = i+1

        return explan_feat

    def process_data(self, outcome_keys): #Todo: add more options for explan_feat types

        # Verify all the neurons are the same for all the trials
        roiNames = self.data['imagingData']['roiNames'][0][0]
        # Verify all the columns are the same
        if not np.all(np.all(roiNames[:, 1:] == roiNames[:, :-1], axis=0)):
            raise ValueError("roiNames are not the same for all trials")
        # Find indices in roiNames where values match the relevant_idx
        relative_idx_neu = np.array(np.where(np.isin(roiNames[:, 0], self.relevent_idx)))
        if self.data is None:
            raise ValueError("Data not loaded properly.")

        # Init parameters
        samples_num = self.data['imagingData']['samples'][0][0].shape[1]
        if not samples_num == self.sample_per_sec*(self.end_time-self.start_time):
            if samples_num == 600:
                self.end_time = 16
                assert samples_num == self.sample_per_sec*(self.end_time-self.start_time)
            else:
                raise ValueError(" total number of samples number is not consistence")
        trial_num = self.data['imagingData']['samples'][0][0].shape[2]

        self.explan_feat = self.get_explan_feat(relative_idx_neu, np.int(self.window_size_avg * self.sample_per_sec),
                                                np.int(self.overlap_avg * self.sample_per_sec), samples_num)

        # Temp calculations
        eff_t_len = np.int(self.explan_feat.shape[0]/trial_num)
        t = np.linspace(self.start_time + self.drop_time, self.end_time, eff_t_len)
        tmp_output_labels = self.get_output_labels(outcome_keys)
        assert np.unique(tmp_output_labels).shape[0] == 2, "This version of the code does not support multiclass"
        trial_inds = np.arange(1, trial_num + 1)

        self.time_win = np.repeat(t, trial_num)[:, None]
        self.context_feat = stats.zscore(self.time_win)  # (time_bin-mu(time_bin))/sigma(time_bin)
        self.output_label = np.tile(tmp_output_labels, eff_t_len)[:, None]
        self.trials = np.tile(trial_inds, eff_t_len)[:, None]


class AnimalDataProcessor:
    def __init__(self, params):

        self.info_excel_path = params.info_excel_path
        self.mat_files_directory = params.mat_files_directory
        self.animal_info_df = self.read_exel_info(params.info_excel_path, sheet_num=params.sheet_num)  # excel in dataframe
        self.idx_neurons_all_dates = self.find_neurons_intersection()

        # Initialization of all Dates data processor. option to combine the data from a different dates.
        # Todo: add combination
        matfile_path = os.path.join(os.path.join(self.mat_files_directory, params.date), 'data.mat')
        data_tmp = DateProcessor(params, matfile_path, relevent_idx=self.idx_neurons_all_dates,
                                 date_info=self.animal_info_df[self.animal_info_df['folder'] == matfile_path.split('\\')[-2]])
        data_tmp.process_data(outcome_keys=params.outcome_keys)
        if not params.post_process_mode:
            with open(os.path.join(params.res_directory, 'log.txt'), 'a') as f:
                # Add a chance level parameter to the log file
                f.write("%s = %s\n" % ('end_time', data_tmp.end_time))
        self.use_flag = data_tmp.use_flag
        if not self.use_flag:
            return

        # The combined data from all chosen dates
        self.explan_feat = data_tmp.explan_feat  # after multiply by contextual gates will be input to the prediction model
        self.context_feat = data_tmp.context_feat  # inputs to the hyper-network that creates mu vector
        self.output_label = data_tmp.output_label  # labels
        chance_level = self.output_label.sum()/self.output_label.shape[0]
        chance_level = max(chance_level, 1-chance_level)  # Todo: error for multiclass
        params.chance_level = chance_level
        if not params.post_process_mode:
            with open(os.path.join(params.res_directory, 'log.txt'), 'a') as f:
                # Add a chance level parameter to the log file
                f.write("%s = %s\n" % ('chance_level', chance_level))
        print(f"chance level is {chance_level}")
        self.trials = data_tmp.trials
        self.num_trials = self.trials[-1][0]

        self.foldsnum = params.folds_num
        self.traininds = []
        self.devinds = []
        self.testinds = []
        if not params.post_process_mode:  # finding hyperparameters
            self.split_data_into_folds(num_trials=self.num_trials)
        else:  # post process, after the hyperparameters is chosen
            self.split_train_test(self.num_trials)

    def read_exel_info(self, info_excel_path, sheet_num):
        # Read data information from the Excel file
        date_info_df = pd.read_excel(self.info_excel_path, sheet_name=sheet_num)
        return date_info_df

    def find_neurons_intersection(self):

        first_flag = True
        # Iterate through all "data.mat" files for all dates for a specific animal
        for file in os.listdir(self.mat_files_directory):
            condition = (self.animal_info_df['folder'] == file)
            include_num = self.animal_info_df.loc[condition, 'to_include'].values[0]
            if include_num == 2:  # 2 - include date, 1 - maybe, 0 - not include
                # Find common numbers across all columns in imagingData.roiNames
                mat_path = os.path.join(os.path.join(self.mat_files_directory, file), 'data.mat')
                data = spio.loadmat(mat_path)
                roi_names = data['imagingData']['roiNames'][0][0]
                common_numbers = set(roi_names[:, 0])
                for col in range(1, roi_names.shape[1]):
                    common_numbers.intersection_update(roi_names[:, col])

                # Find common numbers across all dates
                if first_flag:
                    idx_neurons_all_dates = set(common_numbers)
                    first_flag = False
                else:
                    idx_neurons_all_dates.intersection_update(common_numbers)

        return np.array(list(idx_neurons_all_dates))

    def split_data_into_folds(self, num_trials):

        kf = KFold(n_splits=self.foldsnum)
        kf.get_n_splits(range(num_trials))

        for traindev_trials, test_trials in kf.split(range(num_trials)):

            train_inds, dev_inds = train_test_split(traindev_trials, test_size=0.2, shuffle=True)

            train = self.trials2inds(train_inds, (self.trials - 1))
            dev = self.trials2inds(dev_inds, (self.trials - 1))
            test = self.trials2inds(test_trials, (self.trials - 1))

            self.traininds.append(train)
            self.devinds.append(dev)
            self.testinds.append(test)

    def split_train_test(self, num_trials):

        train_inds, test_inds = train_test_split(range(num_trials), test_size=0.2, shuffle=True)

        train = self.trials2inds(train_inds, (self.trials - 1))
        test = self.trials2inds(test_inds, (self.trials - 1))

        self.traininds.append(train)
        self.testinds.append(test)

    def trials2inds(self, sel_trials, trials_labels):
        # return all the indices according to self.trials that suitable for the trials number in the fold
        trials_inds = np.zeros_like(trials_labels)
        for trial in sel_trials:
            trials_inds[trials_labels == trial] = True
        return [i for i, x in enumerate(trials_inds) if x] # return all the indices of the element in trials_inds that are not zero

    def save_processed_data(self, output_filename):  # Todo: change the save parameters
        if self.X is None or self.outcome_label is None or self.time_win is None\
                or self.trials is None or self.internal_param is None:
            raise ValueError("Data not processed. Call process_data() first.")

        spio.savemat(output_filename, {'X': self.X, 'outcome_label': self.outcome_label, 'time_win': self.time_win,
                                       'trials': self.trials, 'internal_param': self.internal_param})


class DataProcessor(AnimalDataProcessor):
    def __init__(self, params):
        AnimalDataProcessor.__init__(self, params)


class DataContainer:
    def __init__(self, params, data, fold):

        # train set
        self.xtr = data.explan_feat[data.traininds[fold], :]
        self.ytr = data.output_label[data.traininds[fold]]
        self.rtr = data.context_feat[data.traininds[fold]]
        self.rtr = 2 * (norm_minmax(self.rtr.reshape(-1, 1)) - 0.5)
        # test set
        self.xte = data.explan_feat[data.testinds[fold], :]
        self.yte = data.output_label[data.testinds[fold]]
        self.rte = data.context_feat[data.testinds[fold]]
        self.rte = 2 * (norm_minmax(self.rte.reshape(-1, 1)) - 0.5)
        # develop set
        if not params.post_process_mode:  # finding hyperparameters
            self.xdev = data.explan_feat[data.devinds[fold], :]
            self.ydev = data.output_label[data.devinds[fold]]
            self.rdev = data.context_feat[data.devinds[fold]]
            self.rdev = 2 * (norm_minmax(self.rdev.reshape(-1, 1)) - 0.5)

        # train
        xtr = self.xtr
        ytr = self.ytr[:, None] if len(self.ytr.shape) == 1 else self.ytr  # one hot
        rtr = self.rtr[:, None] if len(self.rtr.shape) == 1 else self.rtr
        # test
        xtest = self.xte
        ytest = self.yte[:, None] if len(self.yte.shape) == 1 else self.yte
        rtest = self.rte[:, None] if len(self.rte.shape) == 1 else self.rte
        # develop
        if not params.post_process_mode:  # finding hyperparameters
            xdev = self.xdev
            ydev = self.ydev[:, None] if len(self.ydev.shape) == 1 else self.ydev
            rdev = self.rdev[:, None] if len(self.rdev.shape) == 1 else self.rdev

            ytest = torch.empty_like(torch.tensor(rtest))

        # Datasets
        train_set = data_utils.TensorDataset(torch.tensor(xtr), torch.tensor(ytr), torch.tensor(rtr))
        if not params.post_process_mode:  # finding hyperparameters
            dev_set = data_utils.TensorDataset(torch.tensor(xdev), torch.tensor(ydev), torch.tensor(rdev))
            test_set = data_utils.TensorDataset(torch.tensor(xtest), ytest, torch.tensor(rtest))
        else:
            test_set = data_utils.TensorDataset(torch.tensor(xtest), torch.tensor(ytest), torch.tensor(rtest))

        # Dataloaders
        self.train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=params.batch_size, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=params.batch_size, shuffle=False)
        if not params.post_process_mode:  # finding hyperparameters
            self.dev_dataloader = torch.utils.data.DataLoader(dev_set, batch_size=params.batch_size, shuffle=True)

    def get_Dataloaders(self, params):
        if not params.post_process_mode:  # finding hyperparameters
            return self.train_dataloader, self.dev_dataloader, self.test_dataloader
        else:
            return self.train_dataloader, self.test_dataloader



