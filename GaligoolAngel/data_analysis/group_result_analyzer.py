import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from GaligoolAngel.data_analysis.result_analyzer import ResultAnalyzer

class GroupResultAnalyzer:
    def __init__(self, neurons, subsets_names, cc):
        """Initializes an instance of the class.

        Args:
            results (DataFrame): a DataFrame of ResultAnalyzer objects to perform analysis on, each group has a name.
        """
        data = {"neurons": neurons, "subsests": subsets_names}
        if all(isinstance(item, ResultAnalyzer) for item in neurons):
            self.neurons = []
            for analyzer in neurons:
                self.neurons.append(analyzer.neurons)
        else:
            self.neurons = neurons
        self.subsets = subsets_names
        self.cc = cc # The overall correlation matrix
        
        # Analysis Results Properties
        self.corrs = []
        
    def corr_analysis(self, names):
        num_corrs = int(1/2 * (len(names) - 1) * len(names)) # The number of different correlations by the number of subsets given.
        corrs = np.ones((num_corrs, num_corrs)) # Creating a correlation matrix
        for n in range(len(self.neurons)): # Each element is a np array of indices
            for k in range(len(self.neurons) - n):
                corrs[n, k] = np.mean(np.corr(self.cc[:, self.neurons[n]], self.cc[:, self.neurons[k]]))
        
            