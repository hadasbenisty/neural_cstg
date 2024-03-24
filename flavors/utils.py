import numpy as np
from sklearn.metrics import accuracy_score
import os
import torch
import torch.nn as nn
import random
from analysis_results.utils import find_best_hyper_comb


def acc_score(targets, prediction, params):
    if params.classification_flag and params.output_dim == 1:
        acc = accuracy_score(targets, np.int64((prediction.reshape((1, -1)) > 0.5).reshape(-1, 1)))
    elif params.classification_flag and params.output_dim > 2:
        _, predicted_labels = torch.max(prediction, 1)
        correct_predictions = (predicted_labels == targets.flatten()).sum().item()
        total_predictions = targets.size(0)
        acc = correct_predictions / total_predictions
    else:
        raise ValueError("Not supported in this code version")
    return acc


def get_subdirectories(directory):
    subdirectories = [d for d in os.listdir(directory)
                      if os.path.isdir(os.path.join(directory, d)) and not d.endswith('.png') and not d.endswith('.mat')]
    return subdirectories


def init_criterion(param):
    if param.output_dim == 1:
        criterion = nn.BCELoss()
    elif param.output_dim > 2:
        criterion = criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Init criterion problem")
    return criterion


def init_optimizer(model, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    return optimizer


def hyperparameters_chosen_extraction(infer_directory):
    # Extract the best hyperparameters
    best_acc_dev_folder = find_best_hyper_comb(infer_directory, key="nn_acc_dev")
    hyper_hidden_dim_idx = best_acc_dev_folder.find('hidden') + len('hidden') + 1
    hyper_hidden_dim = [int(best_acc_dev_folder[hyper_hidden_dim_idx:].split(']')[0])]
    hidden_dim = [[500, 300, 100, 50, 10, 2]] #params.hidden_dims[0]  # only one option
    learning_rate_idx = best_acc_dev_folder.find('lr') + len('lr')
    learning_rate = float(best_acc_dev_folder[learning_rate_idx:].split('_')[0])
    stg_regularizer = 0.5#params.stg_regularizers[0]  # lambda, only one option
    hyperparameter_combination = 'c-stg' + "_hidden" + str(hyper_hidden_dim) + "_lr" + \
                                 str(learning_rate) + "_lam" + str(stg_regularizer)
    filename = os.path.join(infer_directory, hyperparameter_combination + "_Final_check" + ".mat")

    return filename, hidden_dim, hyper_hidden_dim, learning_rate, stg_regularizer, hyperparameter_combination


def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

