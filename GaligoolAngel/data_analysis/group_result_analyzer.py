from os import name
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from GaligoolAngel.data_analysis.result_analyzer import ResultAnalyzer
from GaligoolAngel.utils import calculate_correlations_columns
import os
class GroupResultAnalyzer:
    def __init__(self, neurons, subsets_names, cc, session_partition):
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
        self.session_parition = session_partition
        
    

    def corr_analysis(self, names):
        # Initialize a list to store correlation values, since the output shape is dynamic
        corrs = np.zeros((len(names), len(names), self.cc.shape[-1]))

        # Map names to indices in self.neurons for easier reference
        name_to_index = {name: index for index, name in enumerate(self.subsets)}

        # Prepare a structured output, a dictionary that can hold variable-length correlation arrays
        corrs_dict = {}

        # Compute correlations for each pair of names
        for trial in range(self.cc.shape[-1]):
            for i, name1 in enumerate(names):
                for j, name2 in enumerate(names[i:], i):
                    n = name_to_index[name1]
                    k = name_to_index[name2]

                    # Extract the columns for the two sets of neurons
                    data1 = self.cc[:, self.neurons[n], trial]
                    data2 = self.cc[:, self.neurons[k], trial]

                    # Initialize an empty matrix to store pairwise correlations
                    # The size depends on the number of columns in data1 and data2
                    temp_corrs = np.zeros((data1.shape[1], data2.shape[1]))

                    # Iterate over each column in data1 and data2 to compute pairwise correlations
                    for col1 in range(data1.shape[1]):
                        for col2 in range(data2.shape[1]):
                            # Calculate the correlation coefficient for the current column pair
                            # np.corrcoef returns a 2x2 matrix, [0,1] or [1,0] contains the correlation coefficient
                            corr_matrix = np.corrcoef(data1[:, col1], data2[:, col2])
                            temp_corrs[col1, col2] = corr_matrix[0, 1]

                    # Store the matrix of correlations for this pair of names
                    corrs[i, j, trial] = np.mean(temp_corrs)
                    corrs[j, i, trial] = corrs[i, j, trial] # symmetric

        self.corrs = corrs
        return corrs
    
    def plot_results(self, path, names):
        # Calculate the number of subplots needed
        num_names = len(names)
        num_corrs = int(num_names * (num_names - 1) / 2)
        
        # Determine the layout of the subplots (trying to make it as square as possible)
        num_rows = int(np.ceil(np.sqrt(num_corrs)))
        num_cols = int(np.ceil(num_corrs / num_rows))

        # Create a figure with subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 8))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust space between plots

        # Flatten the axs array for easy iteration, in case of multiple rows and cols
        axs = axs.ravel()

        # Keep a counter for the current subplot
        subplot_idx = 0

        # Plot correlation for each pair of names
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i < j:  # Ensure we only plot each pair once
                    # Extract the indices for easier referencing
                    n = self.subsets.index(name1)
                    k = self.subsets.index(name2)
                    
                    # Plot correlation for this pair across trials
                    axs[subplot_idx].plot(self.corrs[n, k, :])
                    axs[subplot_idx].set_title(f'Corr {name1} vs {name2}')
                    axs[subplot_idx].set_xlabel('Trials')
                    axs[subplot_idx].set_ylabel('Correlation')
                    
                    subplot_idx += 1

        # Remove unused subplots if any
        for idx in range(subplot_idx, num_rows*num_cols):
            fig.delaxes(axs[idx])
        
        # Save the figure
        plt.tight_layout()
        if not os.path.exists(path):
                os.makedirs(path)
        plt.savefig(os.path.join(path, 'correlation_results.png'))

        # Optionally, close the figure to free memory
        plt.close(fig)
        
        
            