# Imports
import re
import os
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def extract_value(file_path, key, occurrence=1):
    """
    Extracts a specific occurrence of a key from a text file.

    :param file_path: Path to the text file.
    :param key: The key to search for in the file (e.g., "end_time").
    :param occurrence: The occurrence number of the key (1 for first, 2 for second, etc.).
    :return: The value of the specified occurrence of the key, or None if not found.
    """
    with open(file_path, 'r') as file:
        content = file.read()

        # Regex pattern to find the key and its value
        if key == "flavors2num" or key == "animal2sheet_num":
            pattern = rf"{key} = (.+)"
        else:
            pattern = rf"{key} = ([\w._-]+)"
        matches = re.findall(pattern, content)

        if len(matches) >= occurrence:
            return matches[occurrence - 1]
        else:
            return None

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


def plot_aligned_data_statistics(main_parts, acc_per_time_each_part, figure_title, save_path, y_label="Accuracy", combined_plot=False):

    # Filter out empty sublists and their corresponding titles
    non_empty_parts = [(part, title) for part, title in zip(acc_per_time_each_part, main_parts) if part]

    if combined_plot:
        # Setup one plot for all graphs
        fig, ax = plt.subplots(figsize=(10, 5))  # Adjust size as needed

    else:
        # Setup the plot with the number of non-empty sublists
        fig, axs = plt.subplots(1, len(non_empty_parts), figsize=(5 * len(non_empty_parts), 5))
        # Ensure axs is iterable when there's only one subplot
        if len(non_empty_parts) == 1:
            axs = [axs]

    # Define different colors and line styles for each graph
    colors = ['royalblue', 'forestgreen', 'darkorange', 'crimson']
    line_styles = ['-', '--', '-.', ':']

    for idx, (part_data, title) in enumerate(non_empty_parts):
        # Setup ax depending on whether it is a combined plot or separate subplots
        ax = ax if combined_plot else axs[idx]

        # Determine the maximum length of vectors in this part
        max_len = max(len(vec) for vec in part_data)
        start_time = -3 + 0.5
        end_time = 8-0.5 if max_len == 21 else 16-0.5

        # Initialize an array of zeros of shape (number of vectors, max_len)
        #aligned_data = np.zeros((len(part_data), max_len))
        aligned_data = np.full((len(part_data), max_len), np.nan)

        # Fill the aligned_data with vectors, aligned at the start
        for i, vec in enumerate(part_data):
            aligned_data[i, :len(vec)] = vec

        # Calculate the mean and standard error
        masked_data = np.ma.masked_invalid(aligned_data)
        mean_vector = np.ma.mean(masked_data, axis=0).filled(np.nan)  # Compute mean, replace masked with np.nan
        std_error = np.ma.std(masked_data, axis=0, ddof=1) / np.sqrt(masked_data.count(axis=0))  # Compute std error

        # Plotting
        time = np.linspace(start_time, end_time, max_len)
        ax.plot(time, mean_vector, line_styles[idx % len(line_styles)], linewidth=2, label=title, color=colors[idx % len(colors)])
        ax.fill_between(time, mean_vector - std_error, mean_vector + std_error, color=colors[idx % len(colors)], alpha=0.3)
        ax.axvline(x=0, color='red', linestyle='--')
        if not combined_plot:
            ax.set_title(title)
            #ax.set_ylim([0, 1])
            ax.set_xlabel("Time[sec]")
            ax.set_ylabel(y_label)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Additional settings for the combined plot
    if combined_plot:
        ax.set_title(figure_title)
        #ax.set_ylim([0.3, 0.6])
        ax.set_xlabel("Time[sec]")
        ax.set_ylabel(y_label)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Common settings and save the figure
    fig.suptitle(figure_title if not combined_plot else '')
    plt.tight_layout()
    plt.savefig(save_path)

def extract_sortable_date(subdir):
    # Splitting by underscore and extracting the date part
    parts = subdir.split('_')
    date_part = (parts[-4] + '_' + parts[-3] + '_' + parts[-2])  # Assuming the date is always at the end
    # Converting date format from MM_DD_YY to a sortable format YYYYMMDD
    month, day, year = date_part.split('_')
    sortable_date = '20' + year + month + day  # Converting to 'YYYYMMDD' format for sorting
    return sortable_date

def compare_matrices_columns(mat1, mat2):

    # Number of columns (assuming both matrices have the same shape)
    num_columns = mat1.shape[1]

    # Results storage
    results = {
        'only_mat1': [],
        'only_mat2': [],
        'both': []
    }

    # Iterate through columns
    for i in range(num_columns):
        vec1 = mat1[:, i]
        vec2 = mat2[:, i]

        # Calculate conditions
        only_mat1 = np.mean((vec1 >= 0.5) & (vec2 < 0.5))
        only_mat2 = np.mean((vec2 >= 0.5) & (vec1 < 0.5))
        both = np.mean((vec1 >= 0.5) & (vec2 >= 0.5))

        # Store results
        results['only_mat1'].append(only_mat1)
        results['only_mat2'].append(only_mat2)
        results['both'].append(both)

    return np.array(results['only_mat1']), np.array(results['only_mat2']), np.array(results['both'])

def extract_from_log(date_directory, info_excel_path, include_chance_level,
                                      animal2sheet_num, flavors2num, time_point):

    # Extract parameters from the name of the directory
    subdir = date_directory.split('//')[0]
    date = subdir.split('_')[-4:-1]
    date = '_'.join(date)
    animal = subdir.split('_')[-6]
    animal_info_df = pd.read_excel(info_excel_path, sheet_name=animal2sheet_num[animal])
    type1 = animal_info_df[animal_info_df['folder'] == date]['type1'].iloc[0]
    type2 = animal_info_df[animal_info_df['folder'] == date]['type2'].iloc[0]

    # Extract parameters from log file
    if include_chance_level:
        log_file = os.path.join(subdir, "log.txt")
        chance_level = float(extract_value(log_file, "chance_level", occurrence=1))
    else:
        chance_level = 0
    log_file = os.path.join(date_directory, "log.txt")
    start_time = float(extract_value(log_file, "start_time", occurrence=1))
    drop_time = float(extract_value(log_file, "drop_time", occurrence=1))
    start_time = start_time + drop_time
    end_time = float(extract_value(log_file, "end_time", occurrence=2))
    num_time_points = 21 if end_time == 8 else 37
    time_vals = np.linspace(start_time + 0.5, end_time - 0.5, num_time_points)
    if time_point is not None:
        differences = np.abs(time_vals - time_point)
        time_idx = np.argmin(differences)  # Find the index of the smallest difference
    else:
        time_idx = None
    #flavors2num_tmp = ast.literal_eval(extract_value(log_file, "flavors2num", occurrence=1))
    #animal2sheet_num = ast.literal_eval(extract_value(log_file, "animal2sheet_num", occurrence=1))
    info_df = pd.read_excel(info_excel_path, sheet_name=animal2sheet_num[animal])
    flavors_list = info_df[info_df['folder'] == date]['flavors'].iloc[0].split('_')
    # Sort flavors based on the values in the flavors2num dictionary
    for i, element in enumerate(flavors_list):
        if element == 'r':
            flavors_list[i] = 'g'

    flavors = sorted(flavors_list, key=lambda flavor: flavors2num[flavor])

    return animal, date, type1, type2, chance_level, time_vals, flavors, time_idx

def combine_list_of_conf_mat(matrics_list, labels2num):
    # Initialize a master matrix and counts (adding a layer for sum of squares)
    master_matrix = np.zeros((len(labels2num), len(labels2num), 3))  # Last dimension for sum, count, sum of squares

    # Aggregate matrices
    for matrix, labels in matrics_list:
        if matrix.shape[0] != len(labels):
            print("pass")  # there are no all the labels at this specific time context
            continue
        indices = [labels2num['g']-1 if label == 'r' else labels2num[label]-1 for label in labels]
        #indices = [labels2num[label] for label in labels]
        for i, label_i in enumerate(indices):
            for j, label_j in enumerate(indices):
                value = matrix[i, j]
                master_matrix[label_i, label_j, 0] += value  # Sum
                master_matrix[label_i, label_j, 1] += 1  # Count
                master_matrix[label_i, label_j, 2] += value ** 2  # Sum of squares


    # Calculate the mean and std matrix
    mean_matrix = np.zeros((len(labels2num), len(labels2num)))
    err_matrix = np.zeros((len(labels2num), len(labels2num)))

    for i in range(len(labels2num)):
        for j in range(len(labels2num)):
            count = master_matrix[i, j, 1]
            if count > 0:  # Avoid division by zero
                mean = master_matrix[i, j, 0] / count
                sum_of_squares = master_matrix[i, j, 2]
                mean_square = sum_of_squares / count
                variance = max(mean_square - mean ** 2, 0)
                std = np.sqrt(variance)
                error = std / np.sqrt(count)

                mean_matrix[i, j] = mean
                err_matrix[i, j] = error

    return mean_matrix, err_matrix


def plot_1_open_gate(sub_idx, ax_idx, i, chosen, data, gs, time_values, mu_vals):
    if sub_idx % 2 == 0 and sub_idx != 0:
        sub_idx = 0
        ax_idx += 1
    neu_idx = chosen[i]  # opened_neuron_indices[i+20]

    # mus vs time per neuron
    ax = plt.subplot(gs[ax_idx])
    ax_idx += 1
    sub_idx += 1
    ax.plot(time_values, mu_vals[neu_idx, :])
    ax.axvline(x=0, color='red', linestyle='--')
    ax.set_xlabel("Time [sec]", fontsize=14)
    ax.set_ylabel(f"Cell weight {i}", fontsize=14)


    # mean activity
    ax = plt.subplot(gs[ax_idx])
    ax_idx += 1
    sub_idx += 1
    activities, errors = data.get_neu_activity(neu_idx)
    labels = ["success trials", "failure trials"]
    colors = ["green", "orange"]
    for neu_activity, error, lab, color in zip(activities, errors, labels, colors):
        if ax_idx == 5:
            ax.plot(time_values, neu_activity, label=lab, color=color)
            ax.legend(fontsize=14)  # Increase the fontsize for the legend
        else:
            ax.plot(time_values, neu_activity, color=color)
        ax.axvline(x=0, color='red', linestyle='--')
        ax.fill_between(time_values, neu_activity - error, neu_activity + error, alpha=0.5,
                        color=color)  # edgecolor='#CC4F1B', facecolor='#FF9848')

    ax.set_xlabel("Time [sec]", fontsize=14)
    ax.set_ylabel("Activity", fontsize=14)

    return sub_idx, ax_idx

def init_date2idx(animal, cotext=''):
    # Time context - relevant dates
    if cotext == 'time':
        if animal == '4458':
            date2idx = {'22.01.19': 0, '24.01.19': 1, '28.01.19': 2, '24.02.19': 3}
        elif animal == '4575':
            date2idx = {'05.03.19': 0, '10.03.19': 1, '14.03.19': 2, '19.03.19': 3, '31.03.19': 4, '03.04.19': 5, '07.04.19': 6,
                        '11.04.19': 7, '15.04.19': 8}
        elif animal == '4754':
            date2idx = {'01.05.19': 0, '14.05.19': 1, '19.05.19': 2, '21.05.19': 3, '23.05.19': 4,
                        '26.05.19': 5, '28.05.19': 6, '30.05.19': 7, '17.06.19': 8, '23.06.19': 9}
        elif animal == '4756':
            # date2idx = {'12.04.19': 0, '14.05.19': 1, '19.05.19': 2, '21.05.19': 3, '23.05.19': 4,
            #             '26.05.19': 5, '28.05.19': 6, '30.05.19': 7, '11.06.19': 8, '13.06.19': 9}
            date2idx = {'14.05.19': 0, '19.05.19': 1, '21.05.19': 2, '23.05.19': 3,
                        '26.05.19': 4, '28.05.19': 5, '30.05.19': 6, '11.06.19': 7, '13.06.19': 8}
        elif animal == '4880':
            date2idx = {'10.07.19': 0, '14.07.19': 1, '16.07.19': 2, '18.07.19': 3, '21.07.19': 4,
                        '23.07.19': 5, '25.07.19': 6}
        elif animal == '4882':
            date2idx = {'03.07.19': 0, '07.07.19': 1, '16.07.19': 2, '01.08.19': 3, '07.08.19': 4, '15.08.19': 5}
    # Flavors context - relevant dates
    if cotext == 'flavors':
        if animal == '4458':
            date2idx = {'28.01.19': 0, '24.02.19': 1}
        elif animal == '4575':
            date2idx = {'14.03.19': 0, '19.03.19': 1, '31.03.19': 2, '03.04.19': 3, '07.04.19': 4,
                        '11.04.19': 5, '15.04.19': 6}
        elif animal == '4754':
            date2idx = {'19.05.19': 0, '21.05.19': 1, '23.05.19': 2,
                        '26.05.19': 3, '28.05.19': 4, '30.05.19': 5, '17.06.19': 6, '23.06.19': 7}
        elif animal == '4756':
            # date2idx = {'12.04.19': 0, '14.05.19': 1, '19.05.19': 2, '21.05.19': 3, '23.05.19': 4,
            #             '26.05.19': 5, '28.05.19': 6, '30.05.19': 7, '11.06.19': 8, '13.06.19': 9}
            date2idx = {'19.05.19': 0, '21.05.19': 1, '23.05.19': 2,
                        '26.05.19': 3, '28.05.19': 4, '30.05.19': 5, '11.06.19': 6, '13.06.19': 7}
        elif animal == '4880':
            date2idx = {'18.07.19': 0, '21.07.19': 1, '23.07.19': 2, '25.07.19': 3}
        elif animal == '4882':
            date2idx = {'16.07.19': 0, '01.08.19': 1, '07.08.19': 2, '15.08.19': 3}

    return date2idx

def visual_correlation_1_animal(animal_directory, animal, all_eff_mu_vals, combined_labels ):
    # Visual results
    plt.figure(figsize=(15, 10))
    ax = sns.heatmap(np.transpose(np.array(all_eff_mu_vals)), annot=False, cmap="YlGnBu", cbar=True)
    plt.title('Changes in Neurons States Across Dates' + '\n' + animal)
    plt.ylabel('Neurons States')
    ax.set_xticklabels(combined_labels, rotation=0)
    plt.savefig(os.path.join(animal_directory, 'neurons_states_across_dates.png'))

    # Change in order for better visualization
    ic = np.argsort(np.transpose(all_eff_mu_vals)[:, -1])
    sorted_mu_eff_per_nue = np.transpose(all_eff_mu_vals)[np.flip(ic), :]
    plt.figure(figsize=(15, 10))
    ax = sns.heatmap(sorted_mu_eff_per_nue, annot=False, cmap="YlGnBu", cbar=True)
    plt.title('Changes in Neurons States Across Dates' + '\n' + animal)
    plt.ylabel('Neurons States')
    ax.set_xticklabels(combined_labels, rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(animal_directory, 'neurons_states_across_dates_organized.png'))

    # Calculate the correlation matrix between each of the dates
    correlation_matrix = np.corrcoef(all_eff_mu_vals)  # Pearson correlation coefficient
    plt.figure(figsize=(15, 10))
    ax = sns.heatmap(correlation_matrix,
                     annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('Correlation Matrix across dates experiments' + '\n' + animal)
    ax.set_xticklabels(combined_labels, rotation=0)
    ax.set_yticklabels(combined_labels, rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(animal_directory, 'Dates_correlation.png'))
    spio.savemat(os.path.join(animal_directory, 'Dates_correlation.mat'),
                 {'correlation_matrix': correlation_matrix,
                  'combined_labels': combined_labels})