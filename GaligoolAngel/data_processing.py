import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import scipy.io as spio
from scipy import stats
import torch.utils.data as data_utils
import torch
import random


class DataProcessor:
    def __init__(self, params, matfile_path):
        # ---- Description ----
        # The function intializes the dataprocessor object.
        # ---- Inputs ----
        # params - a params object.(the params class can be viewed in c_stg folder)
        # matfile_path - a string, contains the path to the data files with the relevant data.
        self.data = spio.loadmat(matfile_path)
        self.data_info = None  # Not relevant to us.
        self.explain_feat = self.data['features']  # As built in the matlab code to sort the data
        self.context_feat = self.data['context']
        self.output_label = self.data['y']

        # Partition
        self.folds_num = params.folds_num
        self.test_inds = []  # test batch
        self.rest_inds = []  # all the indices that are not test
        self.split_rest_test()  # Splits the data into test and rest

        self.train_inds = []  # train batch
        self.dev_inds = []  # validation batch
        self.split_data_into_folds()  # splits the rest of the data into folds_num with train and dev

        # Extra Parameters
        self.start_time = params.start_time
        self.end_time = params.end_time

    def split_data_into_folds(self):
        """
        Splits the dev and train data into folds.


        """
        kf = KFold(n_splits=self.folds_num)

        for train_index, dev_index in kf.split(self.explain_feat[self.rest_inds]):
            self.train_inds.append(train_index)
            self.dev_inds.append(dev_index)

    def split_rest_test(self, train_size=0.8):
        """
        Splits the data into training and test sets after shuffling.
        Changes the object test_inds and rest_inds properties

        Parameters:
        train_size (float): The fraction of the data to be used for training (between 0 and 1).
        """
        if not 0 <= train_size <= 1:
            raise ValueError("Train size must be between 0 and 1")
        data_size = len(self.explain_feat)  # The number of samples we have
        indices = list(range(data_size))
        random.shuffle(indices)

        train_count = int(data_size * train_size)
        train_indices = indices[:train_count]
        test_indices = indices[train_count:]

        self.test_inds = test_indices
        self.rest_inds = train_indices


