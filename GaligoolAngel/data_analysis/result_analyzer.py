import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as spio
from result_processor import ResultProcessor
import networkx as nx
import community as community_louvain

class ResultAnalyzer:
    """
    A class to perform analysis on the results, it is given a subset of features.
    """
    def __init__(self, adjacency_matrix):
        """

        Args:
            adjacency_matrix (ndarray): a tensor whose first two dimensions are equal and the 
            last dimension indicates the process over time
        """
        self.adjacency_matrix = adjacency_matrix # The correaltions of the subset along the learning
        self.g = [] # All the graphs represented in the learning process
        for n in range(adjacency_matrix.shape[-1]):
            self.g.append(nx.Graph(adjacency_matrix[:, :, n]))
        
        self.degs = []
        self.eigen_values = []
        self.modularity = []
        self.num_communities []
        self.centrality_matrix = []
        
    # Graph Analysis
    def degree_analysis(self):
        """
        Performs Degree Analysis on the process given
        """
        for gr in self.g:
            self.degs.append(nx.degree_centrality(gr))
            # Determine the matrix size
        num_graphs = len(self.degs)
        max_nodes = max(max(d.keys()) for d in self.degs) + 1  # +1 because node indices start at 0

        # Initialize the matrix with zeros
        self.centrality_matrix = np.zeros((num_graphs, max_nodes))

        # Fill the matrix
        for i, graph_centrality in enumerate(self.degs):
            for node, centrality in graph_centrality.items():
                self.centrality_matrix[i, node] = centrality
    
    def eigen_values_analysis(self):
        """
        Performs EigenValues Analysis on the process given
        """
        for t in range(self.adjacency_matrix.shape[-1]):
            self.eigen_values.append(np.linalg.eigvals(self.adjacency_matrix[:, :, t]))
    
    def community_analysis(self):
        self.modularity = []
        self.num_communities = []
        for gr in self.g:
            partition = community_louvain.best_partition(gr)
            modularity_t = community_louvain.modularity(partition=partition, G=gr)
            num_communities_t = len(set(partition.values))
            self.modularity.append(modularity_t)
            self.num_communities.append(num_communities_t)

    
    # Plotting Results
    def plot_analysis_results(self, type):
        """Plots the analysis results based on the type given

        Args:
            type (string): the type of the analysis that one wants to plot
            the options are:
            - "all" - plot all the analysis 
            - "degree" - plot degree analysis
            - "eigenvals" - plot eigen values analysis
            - "community" - plt community analysis
        """
        if type == 'all':
            self.plot_degree()
            self.plot_eigen_values()
            self.plot_community()
        elif type == "degree":
            self.plot_degree()
        elif type == 'eigenvals':
            self.plot_eigen_values()
        elif type == 'community':
            self.plot_community()
        else:
            raise("The allowed types are: - 'all', 'community', 'degree', 'eigenvals'")
        
    def plot_degree(self):
        plt.figure()
        plt.imshow(self.centrality_matrix)
        plt.title("The Degree Of The Nodes Along The Process")
        plt.ylabel("Nodes [#]")
        plt.xlabel("Process Time [#]")
    
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

    def plot_eigen_values(self):
        plt.figure()
        plt.plot(self.eigen_values[0, :])
        plt.title("The Biggest EigenValue Along The Process")
        plt.ylabel("EigenValue")
        plt.xlabel("Process [#]")