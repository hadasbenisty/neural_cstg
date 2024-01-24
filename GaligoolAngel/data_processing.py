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
        self.foldsnum = params.foldsnum
        self.train_inds = []  # train batch
        self.test_inds = []  # test batch
        self.dev_inds = []  # validation batch
        self.split_data_into_folds()

        # Extra Parameters
        self.start_time = params.start_time
        self.end_time = params.end_time

    def split_data_into_folds(self, test_size=0.2):

        kf = KFold(n_splits=self.foldsnum)
        self.train_inds, self.dev_inds = train_test_split(test_size=test_size, shuffle=True)



