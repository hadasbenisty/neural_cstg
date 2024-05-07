import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import scipy.io as spio
from scipy import stats

class DateProcessor:
    def __init__(self, params, matfile_path, relevent_idx, date_info):

        self.data = spio.loadmat(matfile_path)
        self.date_info = date_info  # Store additional information about the date
        self.raw_data = []
        self.date_flavors = date_info.iloc[0]['flavors'].split('_')
        self.flavors2num = params.flavors2num
        if date_info.iloc[0]['codition'] == 'control' and date_info.iloc[0]['to_include'] == 2:
                #and date_info.iloc[0]['type1'] == 'ongoing':
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
        self.total_time = params.total_time  # -1 for all the data duration, else number of sec
        self.sample_per_sec = params.sample_per_sec  # samples/sec
        self.window_size_avg = params.window_size_avg  # 1 sec
        self.overlap_avg = params.overlap_avg  # 0.5 sec

        self.conflict = False  # Assume no conflict in date data

    def get_output_labels(self, outcome_keys, eff_t_len):
        if outcome_keys[0] == 'flavors':
            success_array = self.data['BehaveData']['success'][0][0]['indicatorPerTrial'][0][0]
            # 1 for success, 0 for failre
            outcomes = []
            allowed_flavors = []
            for flavor in self.date_flavors: #Todo: add option for 'r' regular
                if flavor == 'g':  # {'g': 1, 's': 2, 'q': 3, 'r': 4, 'f': 5, 'fail': 0}
                    flavor_array = self.data['BehaveData']['grain'][0][0]['indicatorPerTrial'][0][0]
                elif flavor == 's':
                    flavor_array = self.data['BehaveData']['sucrose'][0][0]['indicatorPerTrial'][0][0]
                elif flavor == 'q':
                    flavor_array = self.data['BehaveData']['quinine'][0][0]['indicatorPerTrial'][0][0]
                elif flavor == 'r':
                    flavor_array = self.data['BehaveData']['regular'][0][0]['indicatorPerTrial'][0][0] #TODO
                elif flavor == 'f':
                    flavor_array = self.data['BehaveData']['fake'][0][0]['indicatorPerTrial'][0][0]
                else:
                    raise ValueError("There is no flavor like this")
                num_flavor = self.flavors2num[flavor]
                # take only the successful tries
                flavor_array = num_flavor * np.array(flavor_array) * np.array(success_array)
                outcomes.append(flavor_array)
                allowed_flavors.append(num_flavor)

            outcomes = np.stack(outcomes)
            # Check for conflicts: more than one non-zero value at any position
            conflict_mask = np.sum(outcomes != 0, axis=0) > 1
            if np.any(conflict_mask):
                print("Conflict detected in flavors")
                self.conflict = True

            # Use np.max to combine the arrays, since we know there are no conflicts
            outcomes_par_trial = np.max(outcomes, axis=0)

            # Consider only the successful trials, failure trials are 0
            failure_mask = outcomes_par_trial == 0
            drop_mask = failure_mask | conflict_mask

            # Convert to torch cross-entropy format
            # Map non-None values to consecutive integers starting from 0
            unique_vals = np.unique(outcomes_par_trial[outcomes_par_trial != 0])
            mapping = {v: i for i, v in enumerate(unique_vals)}
            output_labels = np.array([mapping[val] if val in mapping else -1 for val in outcomes_par_trial.flatten()])

            output_labels = np.tile(output_labels.flatten(), eff_t_len)[:, None]
            drop_mask = np.tile(drop_mask.flatten(), eff_t_len)[:, None]

            return output_labels, drop_mask

        else:
            outcomes = []
            for key in outcome_keys:
                outcome_label = self.data['BehaveData'][key][0][0]['indicatorPerTrial'][0][0]
                outcomes.append(outcome_label)

            # Stack the arrays horizontally to create a 2D array
            stacked_outcomes = np.column_stack(outcomes)
            # Create a new array based on the 'And' condition, only trials that satisfy all keys are indicate with '1'
            output_labels = np.all(stacked_outcomes, axis=1).astype(int)
            output_labels = np.tile(output_labels, eff_t_len)[:, None]

            return output_labels, None

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

    def get_context_feat(self, context_key, eff_t_len):

        if context_key == 'time':
            return stats.zscore(self.time_win), None  # (time_bin-mu(time_bin))/sigma(time_bin)

        elif context_key == "flavors":
            context = []
            allowed_flavors = []
            for flavor in self.date_flavors:
                if flavor == 'g':  # {'g': 1, 's': 2, 'q': 3, 'r': 4, 'f': 5, 'fail': 0}
                    flavor_array = self.data['BehaveData']['grain'][0][0]['indicatorPerTrial'][0][0]
                elif flavor == 's':
                    flavor_array = self.data['BehaveData']['sucrose'][0][0]['indicatorPerTrial'][0][0]
                elif flavor == 'q':
                    flavor_array = self.data['BehaveData']['quinine'][0][0]['indicatorPerTrial'][0][0]
                elif flavor == 'f':
                    flavor_array = self.data['BehaveData']['fake'][0][0]['indicatorPerTrial'][0][0]
                elif flavor == 'r':
                    flavor_array = self.data['BehaveData']['regular'][0][0]['indicatorPerTrial'][0][0]
                else:
                    raise ValueError("There is no flavor like this")
                num_flavor = self.flavors2num[flavor]
                flavor_array = num_flavor * np.array(flavor_array)
                context.append(flavor_array)
                allowed_flavors.append(num_flavor)

            context = np.stack(context)
            # Check for conflicts: more than one non-zero value at any position
            conflict_mask_1 = np.sum(context != 0, axis=0) > 1
            if np.any(conflict_mask_1):
                print("Conflict detected in flavors")
                self.conflict = True

            # if np.any(np.sum(context != 0, axis=0) > 1):
            #     raise ValueError("Conflict detected in flavors")

            # Use np.max to combine the arrays, since we know there are no conflicts
            context_per_trial = np.max(context, axis=0)

            # Check if the combined context has values not in allowed_flavors
            conflict_mask_2 = ~ np.isin(context_per_trial, np.array(allowed_flavors))
            if np.any(conflict_mask_2):
                print("Problem with the flavors context")
                self.conflict = True

            # if not np.all(np.isin(context_per_trial[~conflict_mask], np.array(allowed_flavors))):
            #     raise ValueError("Problem with the flavors context")

            # Combine both conflict masks
            conflict_mask_total = conflict_mask_1 | conflict_mask_2

            context = np.tile(context_per_trial.flatten(), eff_t_len)[:, None]
            conflict_mask_total = np.tile(conflict_mask_total.flatten(), eff_t_len)[:, None]

            return context, conflict_mask_total

        else:
            raise ValueError("No suitable context key")


    def process_data(self, outcome_keys, context_key): #Todo: add more options for explan_feat types

        # Verify all the neurons are the same for all the trials
        roiNames = self.data['imagingData']['roiNames'][0][0]
        # Verify all the columns are the same
        if not np.all(np.all(roiNames[:, 1:] == roiNames[:, :-1], axis=0)):
            raise ValueError("roiNames are not the same for all trials")
        # Find indices in roiNames where values match the relevant_idx
        relative_idx_neu = np.array(np.where(np.isin(roiNames[:, 0], self.relevent_idx)))
        if self.data is None:
            raise ValueError("Data not loaded properly.")
        self.raw_data = self.data['imagingData']['samples'][0][0][relative_idx_neu, :, :].squeeze()

        # Init parameters
        samples_num = self.data['imagingData']['samples'][0][0].shape[1]
        if not samples_num == self.sample_per_sec*(self.end_time-self.start_time):
            if samples_num == 600:
                self.end_time = 16
                assert samples_num == self.sample_per_sec*(self.end_time-self.start_time)
            else:
                raise ValueError(" total number of samples number is not consistence")
        trial_num = self.data['imagingData']['samples'][0][0].shape[2]

        if self.total_time != -1:
            if self.total_time * self.sample_per_sec > samples_num:
                raise ValueError(" total time is bigger than exists")
            elif self.total_time < 0:
                raise ValueError(" invalid total time")
            else:
                drop_time_end = self.end_time - (self.start_time + self.drop_time + self.total_time)
                samples_num = samples_num - drop_time_end * self.sample_per_sec

        self.explan_feat = self.get_explan_feat(relative_idx_neu, int(self.window_size_avg * self.sample_per_sec),
                                                int(self.overlap_avg * self.sample_per_sec), samples_num)

        # Temp calculations
        eff_t_len = int(self.explan_feat.shape[0]/trial_num)
        t = np.linspace(self.start_time + self.drop_time, self.end_time, eff_t_len)
        trial_inds = np.arange(1, trial_num + 1)
        self.time_win = np.repeat(t, trial_num)[:, None]
        self.trials = np.tile(trial_inds, eff_t_len)[:, None]

        self.output_label, drop_mask_output = self.get_output_labels(outcome_keys, eff_t_len)

        self.context_feat, drop_mask_context = self.get_context_feat(context_key, eff_t_len)

        # Check if either drop_mask_output or drop_mask_context is None, and handle accordingly
        if drop_mask_output is None:
            drop_mask = drop_mask_context
        elif drop_mask_context is None:
            drop_mask = drop_mask_output
        else:
            # If neither is None, perform logical OR operation
            drop_mask = drop_mask_output | drop_mask_context
        if drop_mask is not None:
            non_drop_mask = ~drop_mask.flatten()
            self.context_feat = self.context_feat[non_drop_mask]
            self.output_label = self.output_label[non_drop_mask]
            self.trials = self.trials[non_drop_mask]
            self.time_win = self.time_win[non_drop_mask]
            self.explan_feat = self.explan_feat[non_drop_mask, :]



class AnimalDataProcessor:
    def __init__(self, params):

        self.info_excel_path = params.info_excel_path
        self.mat_files_directory = params.mat_files_directory
        self.animal_info_df = self.read_exel_info(params.info_excel_path, sheet_num=params.sheet_num)  # excel in dataframe
        self.idx_neurons_all_dates = self.find_neurons_intersection()

        # Initialization of all Dates data processor. option to combine the data from a different dates.
        # Todo: add combination of dates
        matfile_path = os.path.join(os.path.join(self.mat_files_directory, params.date), 'data.mat')
        self.Date_data = DateProcessor(params, matfile_path, relevent_idx=self.idx_neurons_all_dates,
                                 date_info=self.animal_info_df[self.animal_info_df['folder'] == matfile_path.split('\\')[-2]])
        self.Date_data.process_data(outcome_keys=params.outcome_keys, context_key=params.context_key)
        params.end_time = self.Date_data.end_time
        if not params.post_process_mode:
            with open(os.path.join(params.res_directory, 'log.txt'), 'a') as f:
                # Change the end_time parameter in the log file, it will be added again
                f.write("%s = %s\n" % ('end_time', self.Date_data.end_time))
            if self.Date_data.conflict:
                with open(os.path.join(params.res_directory, 'log.txt'), 'a') as f:
                    # Add a conflict parameter to the log file
                    f.write("%s = %s\n" % ('conflict_time', 'Has been a conflict'))

        self.use_flag = self.Date_data.use_flag
        if not self.use_flag:
            self.params = params
            return

        # The combined data from all chosen dates
        self.explan_feat = self.Date_data.explan_feat  # after multiply by contextual gates will be input to the prediction model
        self.context_feat = self.Date_data.context_feat  # inputs to the hyper-network that creates mu vector
        self.output_label = self.Date_data.output_label  # labels
        # Chance level is proportion of the most frequent class
        class_counts = np.bincount(self.output_label.flatten())  # Count the frequency of each class
        most_frequent_class_count = np.max(class_counts)  # Find the most frequent class
        chance_level = most_frequent_class_count / len(self.output_label)
        params.chance_level = chance_level
        if not params.post_process_mode:
            with open(os.path.join(params.res_directory, 'log.txt'), 'a') as f:
                # Add a chance level parameter to the log file
                f.write("%s = %s\n" % ('chance_level', chance_level))
        print(f"chance level is {chance_level}")
        self.trials = self.Date_data.trials
        self.num_trials = len(np.unique(self.trials))

        self.foldsnum = params.folds_num
        self.traininds = []
        self.devinds = []
        self.testinds = []
        if not params.post_process_mode:  # finding hyperparameters
            self.split_data_into_folds(num_trials=self.num_trials)
        else:  # post process, after the hyperparameters is chosen
            self.split_train_test(self.num_trials)

        self.params = params

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
        unique_trials = np.unique(self.trials)

        for traindev_trials, test_trials in kf.split(range(num_trials)):

            train_inds, dev_inds = train_test_split(traindev_trials, test_size=0.2, shuffle=True)

            train = self.trials2inds(unique_trials[train_inds], self.trials)
            dev = self.trials2inds(unique_trials[dev_inds], self.trials)
            test = self.trials2inds(unique_trials[test_trials], self.trials)

            # it list of the sublists for every type of data
            self.traininds.append(train)
            self.devinds.append(dev)
            self.testinds.append(test)

    def split_train_test(self, num_trials):

        train_inds, test_inds = train_test_split(range(num_trials), test_size=0.2, shuffle=True)
        unique_trials = np.unique(self.trials)

        train = self.trials2inds(unique_trials[train_inds], self.trials)
        test = self.trials2inds(unique_trials[test_inds], self.trials)

        self.traininds.append(train)
        self.testinds.append(test)

    def trials2inds(self, sel_trials, trials_labels):
        # return all the indices according to self.trials that suitable for the trials number in the fold
        trials_inds = np.zeros_like(trials_labels)
        for trial in sel_trials:
            trials_inds[trials_labels == trial] = True
        return [i for i, x in enumerate(trials_inds) if x] # return all the indices of the element in trials_inds that are not zero


    def get_neu_activity(self, neuron_index):

        trial_times_activity = self.explan_feat[:, neuron_index]
        num_trial = np.unique(self.trials).shape[0]
        labels = self.output_label[:num_trial]
        trials_vs_time = trial_times_activity.reshape(-1, num_trial).T

        succ_labels = labels == 1
        mean_activity_succ = trials_vs_time[succ_labels.flatten(), :].mean(axis=0)
        stds_succ = trials_vs_time[succ_labels.flatten(), :].std(axis=0)/np.sqrt(num_trial-1)  # standard error of the mean, SEM

        fail_labels = labels == 0
        mean_activity_fail = trials_vs_time[fail_labels.flatten(), :].mean(axis=0)
        stds_fail = trials_vs_time[fail_labels.flatten(), :].std(axis=0)/np.sqrt(num_trial-1)  # standard error of the mean, SEM

        mean_activities = [mean_activity_succ, mean_activity_fail]
        stds = [stds_succ, stds_fail]

        return mean_activities, stds


class DataProcessor(AnimalDataProcessor):
    def __init__(self, params):
        AnimalDataProcessor.__init__(self, params)






