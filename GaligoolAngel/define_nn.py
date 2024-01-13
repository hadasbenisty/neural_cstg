import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from variables import output_dim, feature_dim

class Hypernetwork(nn.Module):
    def __init__(self, feature_dim):
        super(Hypernetwork, self).__init__()
        # Define layers of the hypernetwork
        self.weights = nn.Parameter(torch.ones(feature_dim))

    def forward(self, features):
        return features * torch.max(torch.zeros(self.weights.size()),
                                    torch.min(torch.ones(self.weights.size()), self.weights))


class PredictionNetwork(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(PredictionNetwork, self).__init__()
        # Define layers of the prediction network
        self.fc1 = nn.Linear(feature_dim, 64)  # Adjust the sizes as needed
        self.fc2 = nn.Linear(64, output_dim)   # Output dimension depends on your task

    def forward(self, features):
        x = F.relu(self.fc1(features))
        x = self.fc2(x)
        return x

class CSTGModel(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(CSTGModel, self).__init__()
        self.hypernetwork = Hypernetwork(feature_dim)
        self.prediction_network = PredictionNetwork(feature_dim, output_dim)

    def forward(self, features):
        # Get feature selection probabilities from hypernetwork
        features = self.hypernetwork.forward(features)

        # Get predictions from the prediction network
        predictions = self.prediction_network(features)
        return predictions



