import numpy as np
from sklearn.metrics import accuracy_score
import os
import torch
import torch.nn as nn
import scipy.io as spio


def acc_score(targets, prediction):
    acc = accuracy_score(targets, np.int64((prediction.reshape((1, -1)) > 0.5).reshape(-1, 1)))
    return acc


def get_subdirectories(directory):
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return subdirectories


def init_criterion():
    criterion = nn.BCELoss()
    return criterion


def init_optimizer(model, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    return optimizer


def norm_minmax(x):
    if len(np.unique(x)) == 1:
        return x
    m = np.min(x)
    M = np.max(x)
    x = (x - m) / (M - m)
    return x


def hyperparameters_chosen_extraction(params):
    # Extract the best hyperparameters
    best_acc_dev_folder = find_best_hyper_comb(params.infer_directory, key="nn_acc_dev")
    hyper_hidden_dim_idx = best_acc_dev_folder.find('hidden') + len('hidden') + 1
    hyper_hidden_dim = [int(best_acc_dev_folder[hyper_hidden_dim_idx:].split(']')[0])]
    hidden_dim = params.hidden_dims[0]  # only one option
    learning_rate_idx = best_acc_dev_folder.find('lr') + len('lr')
    learning_rate = float(best_acc_dev_folder[learning_rate_idx:].split('_')[0])
    stg_regularizer = params.stg_regularizers[0]  # lambda, only one option
    hyperparameter_combination = params.strfile + "_hidden" + str(hyper_hidden_dim) + "_lr" + \
                                 str(learning_rate) + "_lam" + str(stg_regularizer)
    filename = os.path.join(params.infer_directory, hyperparameter_combination + "_Final_check" + ".mat")

    return filename, hidden_dim, hyper_hidden_dim, learning_rate, stg_regularizer, hyperparameter_combination


def find_best_hyper_comb(root_directory, key):

    # Initialize variables to store the maximum mean and corresponding folder
    max_mean = float('-inf')  # Negative infinity to ensure any mean value will be greater
    best_folder = None

    # Iterate through each sub folder in the root directory
    for subfolder in os.listdir(root_directory):
        subfolder_path = os.path.join(root_directory, subfolder)

        # Check if the current item in the directory is a subfolder
        if os.path.isdir(subfolder_path):
            # Initialize a list to store nn_acc_dev values for the current subfolder
            nn_acc_dev_values = []

            # Iterate through each mat file in the subfolder
            for mat_file in os.listdir(subfolder_path):
                if mat_file.endswith('.mat'):
                    mat_file_path = os.path.join(subfolder_path, mat_file)

                    # Load the mat file and get the nn_acc_dev property
                    mat_data = spio.loadmat(mat_file_path)
                    nn_acc_dev = mat_data.get(key, None)  # return None if the key is not found

                    # Check if nn_acc_dev property exists
                    if nn_acc_dev is not None:
                        # Append the mean value to the list
                        nn_acc_dev_values.append(nn_acc_dev)

            # Calculate the mean of nn_acc_dev values for the current subfolder
            if nn_acc_dev_values:
                subfolder_mean = sum(nn_acc_dev_values) / len(nn_acc_dev_values)

                # Update the maximum mean and corresponding folder if needed
                if subfolder_mean > max_mean:
                    max_mean = subfolder_mean
                    best_folder = subfolder

    # Print the result
    if best_folder is not None:
        print(f"The subfolder with the highest mean {key} is: {best_folder}")
        print(f"The maximum mean value is: {max_mean}")
    else:
        print("No valid subfolders found.")

    return best_folder