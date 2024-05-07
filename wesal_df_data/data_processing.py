import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
import scipy.io as spio
from scipy import stats
import torch.utils.data as data_utils
import torch
import random


class DataProcessor:
    def __init__(self, params):
        # ---- Description ----
        # The function intializes the dataprocessor object.
        # ---- Inputs ----
        # params - a params object.(the params class can be viewed in c_stg folder)
        # matfile_path - a string, contains the path to the data files with the relevant data.
        df = pd.read_csv(params.matfile_path)
        self.data = df.to_numpy()
        # Get column indexes to drop
        columns_to_drop = ['Parent_Reported_Suicidality', 'Parent_Reported_SI', 'Parent_Reported_SB',
                           'Self_Reported_Sl', params.context_feat]
        columns_to_context = [params.context_feat]
        column_indexes_to_drop = [df.columns.get_loc(col) for col in columns_to_drop]
        self.explain_feat = np.delete(self.data, column_indexes_to_drop, axis=1).T.astype(np.float32)

        # todo @Wesal add the columns to here by input , choosing inputs
        column_indexes_to_context = [df.columns.get_loc(col) for col in columns_to_context]
        self.context_feat = df.iloc[:, column_indexes_to_context].to_numpy()
        column_indexes_to_label = [df.columns.get_loc(params.target)]
        self.output_label = self.data[column_indexes_to_label].astype(np.float32)
        self.params = params
        self.use_flag = True

        # Partition
        self.folds_num = params.folds_num
        self.testinds = []  # test batch
        self.restinds = []  # all the indices that are not test
        self.traininds = []  # train batch
        self.devinds = []  # validation batch
        self.split_data_into_folds()
        self.split_dev_train()  # Splits the data into test and rest

        '''testinds_temp = [self.testinds]
        for t in range(self.folds_num - 1):
            testinds_temp.append(self.testinds)
        self.testinds = testinds_temp'''

        # self.split_data_into_folds()  # splits the rest of the data into folds_num with train and dev

        # Extra Parameters
        self.start_time = params.start_time
        self.end_time = params.end_time

    # def split_data_into_folds(self):
    #     """
    #     Splits the dev and train data into folds.
    #
    #
    #     """
    #     output_label_array = self.output_label.values
    #
    #     # Reshape the output label array to add an extra dimension
    #     output_label_reshaped = output_label_array.reshape(-1, 1)
    #     # Step 2: Transpose the explain_feat
    #     explain_feat_transposed = self.explain_feat.transpose()
    #
    #     # Now, the shapes will be compatible
    #     print("Explain_feat shape:", explain_feat_transposed.shape)
    #     print("Output label reshaped shape:", output_label_reshaped.shape)
    #
    #     skf = StratifiedKFold(n_splits=self.folds_num, random_state=None, shuffle=False)
    #
    #     for i,train_index, test_index in enumerate(skf.split(self.explain_feat, output_label_reshaped)):
    #         self.traininds.append(train_index)
    #         self.testinds.append(test_index)
    def split_data_into_folds(self):
        """
        Splits the dev and train data into folds.


        """
        kf = KFold(n_splits=self.folds_num, shuffle=True)

        for train_index, test_index in kf.split(self.explain_feat):
            self.traininds.append(train_index)
            self.testinds.append(test_index)

    def split_dev_train(self, train_size=0.8):
        """
        Splits the data into training and test sets after shuffling.
        Changes the object test_inds and rest_inds properties

        Parameters:
        train_size (float): The fraction of the data to be used for training (between 0 and 1).
        """
        if not 0 <= train_size <= 1:
            raise ValueError("Train size must be between 0 and 1")

        indices = self.traininds
        random.shuffle(indices)
        train_indices = []
        dev_indices = []
        for fold_i in (np.linspace(0, self.folds_num - 1, self.folds_num)).astype(int):
            train_count = int(len(indices[fold_i]) * train_size)
            train_indices.append(indices[fold_i][:train_count])
            dev_indices.append(indices[fold_i][train_count:])

        self.devinds = dev_indices
        self.restinds = train_indices
