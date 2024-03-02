import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import os
import torch
import torch.nn as nn
import scipy.io as spio
import random


def acc_score(targets, prediction, params):
    if params.classification_flag == 'True' and params.output_dim == 1:
        acc = accuracy_score(targets, np.int64((prediction.reshape((1, -1)) > 0.5).reshape(-1, 1)))
    elif params.classification_flag == 'True' and params.output_dim > 2:
        _, predicted_labels = torch.max(prediction, 1)
        correct_predictions = (predicted_labels == targets.flatten()).sum().item()
        total_predictions = targets.size(0)
        acc = correct_predictions / total_predictions
    else:
        acc = mean_squared_error(targets, prediction.reshape(targets.shape)) / 1 # Todo fix np.var(targets)
    return acc


def get_subdirectories(directory):
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return subdirectories


def init_criterion(params):
    # For regression, using Mean Squared Error Loss
    criterion = nn.MSELoss()
    return criterion


def init_optimizer(model, learning_rate):
    reg_strength = 0  # need l2 regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=reg_strength)
    return optimizer


def hyperparameters_chosen_extraction(params):
    # Extract the best hyperparameters
    best_acc_dev_folder = find_best_hyper_comb(params.infer_directory, key="nn_acc_dev")
    hyper_hidden_dim_idx = best_acc_dev_folder.find('hidden') + len('hidden') + 1
    hyper_hidden_dim = [int(x) for x in
                        [best_acc_dev_folder[hyper_hidden_dim_idx:].split(']')[0]][0].split(',')]
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
    min_mean = float('inf')  # Negative infinity to ensure any mean value will be greater
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
                if subfolder_mean < min_mean:
                    min_mean = subfolder_mean
                    best_folder = subfolder

    # Print the result
    if best_folder is not None:
        print(f"The subfolder with the highest mean {key} is: {best_folder}")
        print(f"The min mean value is: {min_mean}")
    else:
        print("No valid subfolders found.")

    return best_folder

#optional if we ever want to check the best hyper comb for seperation and not accuracy
def find_best_hyper_comb_seperation(root_directory, key):
    # Initialize variables to store the maximum mean and corresponding folder
    min_mean = float('inf')  # Negative infinity to ensure any mean value will be greater
    best_folder = None

    # Iterate through each sub folder in the root directory
    for subfolder in os.listdir(root_directory):
        subfolder_path = os.path.join(root_directory, subfolder)

        # Check if the current item in the directory is a subfolder
        if os.path.isdir(subfolder_path):
            # Initialize a list to store nn_acc_dev values for the current subfolder
            mu_lists = []

            # Iterate through each mat file in the subfolder
            for mat_file in os.listdir(subfolder_path):
                if mat_file.endswith('.mat'):
                    mat_file_path = os.path.join(subfolder_path, mat_file)

                    # Load the mat file and get the nn_acc_dev property
                    mat_data = spio.loadmat(mat_file_path)
                    mu_vals = mat_data.get(key, None)  # return None if the key is not found

                    # Check if nn_acc_dev property exists
                    if mu_vals is not None:
                        # Append the mean value to the list
                        mu_lists.append(mu_vals)

            # Calculate the mean of nn_acc_dev values for the current subfolder
            if mu_lists:
                subfolder_mean = sum(mu_lists) / len(mu_lists)

                # Update the maximum mean and corresponding folder if needed
                if seperation_score(subfolder_mean) < min_mean:
                    min_mean = seperation_score(subfolder_mean)
                    best_folder = subfolder

    # Print the result
    if best_folder is not None:
        print(f"The subfolder with the highest mean {key} is: {best_folder}")
        print(f"The min mean value is: {min_mean}")
    else:
        print("No valid subfolders found.")

    return best_folder

def seperation_score(mu_vals):
    for mu in mu_vals:
        score += min(1-mu,mu)^2
    return score

def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def vector_to_symmetric_matrix(vector):
    # Calculate the size of the matrix n using the inverse of the formula for the elements in the upper triangle
    n = int((-1 + np.sqrt(1 + 8*len(vector))) / 2)
    
    # Initialize an n x n matrix filled with zeros
    matrix = np.zeros((n, n))
    
    # Fill in the upper triangle and mirror it to the lower triangle
    index = 0
    for i in range(n):
        for j in range(i, n):
            matrix[i, j] = vector[index]
            matrix[j, i] = vector[index]
            index += 1
            
    return matrix