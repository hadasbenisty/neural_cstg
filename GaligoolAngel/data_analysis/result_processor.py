import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as spio
from GaligoolAngel.utils import vector_to_symmetric_matrix, vector_to_matrix_index

class ResultProcessor:
    """
    A class to take the result from the c-STG and calculate relevant features in the results.
    """
    def __init__(self, result_data_path):
        # Importing Data
        data = spio.loadmat(result_data_path)
        self.feature_weights = data["mu_vals"]
        self.dynamic_features = []
        self.important_features = []
        self.not_important_features = []
        self.dynamic_neurons = []
        self.important_neurons = []
        self.not_important_neurons = []
        self.find_important_features()
        self.find_important_neurons()

    
    def find_important_features(self):
        """
        Separates the features into three subsets:
        - Dynamic Features - Features who's variance along trials is big enough - 
        meaning they are changing through the learning process
        - Important Features - Features who's weights are on average bigger enough - 
        meaning they contribute to the process of the diffusion map
        - Non Important Features - Features who's weights are on average small - 
        meaning they do not contribute to the process of the diffusion map
        """
        feature_var = np.var(self.feature_weights, axis = 1)
        self.dynamic_features = np.where(feature_var > 0.025) # Just a value that is bigger than 0
        feature_mean = np.mean(self.feature_weights, axis=1)
        self.important_features = np.where(feature_mean > 0.025)
        self.not_important_features = np.where(feature_mean <= 0.025)

    def find_important_neurons(self):
        """
        Takes the relevant neurons from the important features
        """
        neurons_num = vector_to_symmetric_matrix(self.feature_weights).shape[0]
        self.dynamic_neurons = vector_to_matrix_index(self.dynamic_features)
        self.important_neurons = vector_to_matrix_index(self.important_features)
        self.not_important_neurons = vector_to_matrix_index(self.not_important_features)
        self.not_important_neurons = np.array(list(set(self.not_important_neurons) - set(self.important_neurons)))
