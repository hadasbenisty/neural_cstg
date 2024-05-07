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
        
        
        # Initialization
        self.explan_feat = None  # after multiply by contextual gates will be input to the prediction model
        self.context_feat = None  # inputs to the hyper-network that creates mu vector
        self.output_label = None  # labels
       
        self.outcome_options = params.outcome_options
        

    def get_output_labels(self, outcome_keys, ..):
        raise ValueError("Implement me")
        return output_labels

    def get_explan_feat(self, var1, var2):
        raise ValueError("Implement me")
        return explan_feat

    def get_context_feat(self, context_key, ..):
        raise ValueError("Implement me")
        return context


    def process_data(self, outcome_keys, ..): 
        raise ValueError("Implement me")
       
        samples_num = ..
        
        self.explain_feat = self.get_explan_feat(..)

       

        self.output_label = self.get_output_labels(..)

        self.context_feat = self.get_context_feat(..)


class TemplateDataProcessor:
    def __init__(self, params):
        # add your specifics, for example:
        self.info_excel_path = params.info_excel_path
        self.mat_files_directory = params.mat_files_directory
        self.animal_info_df = self.read_exel_info(params.info_excel_path, sheet_num=params.sheet_num)  # excel in dataframe
        self.idx_neurons_all_dates = self.find_neurons_intersection()

        # Initialization of all Dates data processor. option to combine the data from a different dates.
        

        # The combined data from all chosen dates
        self.explan_feat = self.Date_data.explain_feat  # after multiply by contextual gates will be input to the prediction model
        self.context_feat = self.Date_data.context_feat  # inputs to the hyper-network that creates mu vector
        self.output_label = self.Date_data.output_label  # labels
        # Chance level is proportion of the most frequent class
        ...
        params.chance_level = chance_level
        if not params.post_process_mode:
            with open(os.path.join(params.res_directory, 'log.txt'), 'a') as f:
                # Add a chance level parameter to the log file
                f.write("%s = %s\n" % ('chance_level', chance_level))
        print(f"chance level is {chance_level}")
        self.trials = self.Date_data.trials
        self.num_trials = self.trials[-1][0]

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

    
    # this is a particular case for flavors project. other data sets might not need this
    def split_data_into_folds(self, num_trials):

        kf = KFold(n_splits=self.foldsnum)
        kf.get_n_splits(range(num_trials))
        # this is a particular case for flavors project. other data sets might not need this
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

       
        self.traininds.append(train_inds)
        self.testinds.append(test_inds)

  

  
class DataProcessor(TemplateDataProcessor):
    def __init__(self, params):
        TemplateDataProcessor.__init__(self, params)






