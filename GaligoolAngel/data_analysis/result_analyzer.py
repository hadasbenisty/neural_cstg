# from cv2 import eigen
from pickletools import read_unicodestring1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as spio
import networkx as nx
import os
# import community as community_louvain

class ResultAnalyzer:
    """
    A class to perform analysis on the results, it is given a subset of features.
    """
    def __init__(self, adjacency_matrix, neurons, session_partition=[]):
        """

        Args:
            adjacency_matrix (ndarray): a tensor whose first two dimensions are equal and the 
            last dimension indicates the process over time
        """
        self.adjacency_matrix = adjacency_matrix # The correaltions of the subset along the learning
        self.neurons = neurons
        self.g = [] # All the graphs represented in the learning process
        if adjacency_matrix.shape[0] != 0:
            for n in range(adjacency_matrix.shape[-1]):
                self.g.append(nx.Graph(adjacency_matrix[:, :, n]))
        
        self.degs_session = []
        self.degs_trial = []
        self.eigen_values = []
        self.modularity = []
        self.num_communities = []
        self.centrality_matrix = []
        self.avg_corr = []
        self.avg_corr_session = []
        self.session_partition = session_partition # A list of lists that consists the time samples for each session (teh length of the list is the number of sessions)
        
    # Correlation Analysis
    def avg_corr_analysis(self, session_partition=None):
        """Calculates the Average Correaltion in the network throught the session.
            If given a session_partition it calculates the average over each session partition

        Args:
            session_partition (list): a list containing at each element a list or array of indices in the last dimension of the adjancy matrix that are relevant for that session
        """
        if self.adjacency_matrix.shape[0] == 0:
            print("No Features To Analyze")
            return
        else:
            if session_partition is None:
                session_partition = self.session_partition # np.range(self.adjacency_matrix.shape[-1])
            for session in session_partition:
                self.avg_corr_session.append(np.mean(np.abs(self.adjacency_matrix[:, :, session])))
                for trial in session:
                    self.avg_corr.append(np.mean(np.abs(self.adjacency_matrix[:, :, trial])))
    # Graph Analysis
    def degree_analysis(self, session_partition=None):
        """
        Performs Degree Analysis on the process given
        """
        if self.adjacency_matrix.shape[0] == 0:
            print("No Features To Analyze")
            return
        else:
            if session_partition is None:
                session_partition = self.session_partition # np.range(self.adjacency_matrix.shape[-1])
            for session in session_partition:
                self.degs_session.append(np.sum(np.mean(np.abs(self.adjacency_matrix[:, :, session]), axis=-1), axis=1))
                for trial in session:
                    self.degs_trial.append(np.sum(np.abs(self.adjacency_matrix[:, :, trial]), axis=1))
        return
            # Determine the matrix size
        '''num_graphs = len(self.degs)
        max_nodes = max(max(d.keys()) for d in self.degs) + 1  # +1 because node indices start at 0

        # Initialize the matrix with zeros
        self.centrality_matrix = np.zeros((num_graphs, max_nodes))

        # Fill the matrix
        for i, graph_centrality in enumerate(self.degs):
            for node, centrality in graph_centrality.items():
                self.centrality_matrix[i, node] = centrality'''
    
    def eigen_values_analysis(self):
        """
        Performs EigenValues Analysis on the process given. If the object has a session partition, it performs the analysis based on sessions.
        """
        if self.adjacency_matrix.shape[0] == 0:
            print("No Features To Analyze")
            return
        self.eigen_values = []
        for t in range(self.adjacency_matrix.shape[-1]):
            self.eigen_values.append(np.sort(np.linalg.eigvalsh(self.adjacency_matrix[:, :, t])))
        self.eigen_values = np.row_stack(self.eigen_values)
        if len(self.session_partition) > 0:
            eigen_values_tmp = []
            for session in self.session_partition:
                eigen_values_tmp.append(np.mean(self.eigen_values[session, :], axis=0))
            self.eigen_values = np.row_stack(eigen_values_tmp)

    '''def community_analysis(self):
        self.modularity = []
        self.num_communities = []
        for gr in self.g:
            partition = community_louvain.best_partition(gr)
            modularity_t = community_louvain.modularity(partition=partition, G=gr)
            num_communities_t = len(set(partition.values))
            self.modularity.append(modularity_t)
            self.num_communities.append(num_communities_t)'''

    
    # Plotting Results
    def plot_analysis_results(self, type, path=None):
        """Plots the analysis results based on the type given

        Args:
            type (string): the type of the analysis that one wants to plot
            the options are:
            - "all" - plot all the analysis 
            - "degree" - plot degree analysis
            - "eigenvalues" - plot eigen values analysis
            - "community" - plt community analysis
            path (string): the path to save the results at
        """
        if self.adjacency_matrix.shape[0] == 0:
            print("No Features To Analyze")
            return
        if type == 'all':
            # Make sure the directory exists, create it if it doesn't
            if not os.path.exists(path):
                os.makedirs(path)

            # Assuming each plotting function returns a matplotlib figure
            fig_deg = self.plot_degree()
            fig_eig = self.plot_eigen_values()
            fig_comm = self.plot_community()
            fig_corr = self.plot_avg_corr()

            # Define filenames for each figure
            filenames = ['degree_distribution.png', 'eigen_values.png', 'community_structure.png', 'average_correlation.png']

            # Save each figure with the corresponding filename
            for fig, filename in zip([fig_deg, fig_eig, fig_comm, fig_corr], filenames):
                fig_path = os.path.join(path, filename)
                fig.savefig(fig_path)

            # Optionally, close the figures after saving to free up memory
            plt.close('all')
            
        elif type == "degree":
            self.plot_degree()
        elif type == 'eigenvals':
            self.plot_eigen_values()
        elif type == 'community':
            self.plot_community()
        else:
            raise("The allowed types are: - 'all', 'community', 'degree', 'eigenvals'")
        
    def plot_degree(self):
        fig = plt.figure()
        plt.imshow(self.centrality_matrix)
        plt.title("The Degree Of The Nodes Along The Process")
        plt.xlabel("Nodes [#]")
        plt.ylabel("Process Time [#]")
        return fig
    
    def plot_community(self):
        # Create a figure with 2 subplots (vertically arranged)
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # figsize is optional

        # Plot number of communities
        axs[0].plot(self.num_communities)
        axs[0].set_title("The Number Of Communities Found Along the Process")
        axs[0].set_xlabel("Process [#]")
        axs[0].set_ylabel("Num Communities [#]")

        # Plot modularity
        axs[1].plot(self.modularity)
        axs[1].set_title("The Modularity of The Graph Along the Process")
        axs[1].set_xlabel("Process [#]")
        axs[1].set_ylabel("Modularity")

        plt.tight_layout()  # Adjust layout to not overlap
        return fig

    def plot_eigen_values(self, heatmap=False):
        if self.eigen_values == []:
            print("No EigenValues")
            return
        fig = plt.figure()
        if len(self.session_partition) > 0:
            if not heatmap:
                for s_i in range(len(self.session_partition)):
                    plt.plot(np.abs(self.eigen_values[s_i, :]), label=f"Session no. {s_i + 1}")
                plt.title("EigenValues Along Sessions")
                plt.xlabel("Neurons [#]")
                plt.ylabel("EigenValues")
                plt.legend()
                
            else:
                plt.imshow(np.abs(self.eigen_values) / np.abs(self.eigen_values[:, 0][:, np.newaxis]))
                plt.xlabel("Neurons [#]")
                plt.ylabel("Sessions [#]")
                plt.colorbar()
        else:
            plt.plot(self.eigen_values[:, 0])
            plt.title("The Biggest EigenValue Along The Process")
            plt.ylabel("EigenValue")
            plt.xlabel("Process [#]")
        return fig
    
    def plot_avg_corr(self):
        # Create a figure and a 1x2 grid of subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # figsize can be adjusted as needed

        # Plot the average correlation in the network during the process on the first subplot
        axs[0].plot(self.avg_corr)
        axs[0].set_title("The Average Correlation in the Network During the Process")
        axs[0].set_ylabel("Correlation")
        axs[0].set_xlabel("Process [#]")

        # Plot the average correlation along the sessions on the second subplot
        axs[1].plot(self.avg_corr_session)
        axs[1].set_title("The Average Correlation Along the Sessions")
        axs[1].set_ylabel("Correlation")
        axs[1].set_xlabel("Sessions [#]")

        # Adjust layout to make room for titles and labels
        plt.tight_layout()
        
        return fig
