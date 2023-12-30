import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def find_indicative_neurons(directory):
    thresholds_list = []
    alpha_eff_per_nue_all = None
    alpha_vals_list = []
    num_dates = 0
    for subdir in os.listdir(directory):
        if not subdir.endswith('.mat') and not subdir.endswith('.png'):
            for file in os.listdir(os.path.join(directory, subdir)):
                if file.endswith('.mat'):
                    mat_path = os.path.join(os.path.join(directory, subdir), file)
                    data = scipy.io.loadmat(mat_path)
                    acc_vals_per_r = data['acc_vals_per_r'][0]
                    alpha_vals = data['alpha_vals']

                    time_values = np.linspace(-4, 16, acc_vals_per_r.shape[0])
                    mean_before_zero = np.mean(acc_vals_per_r[time_values < 0])
                    time_with_enough_acc = np.logical_and(acc_vals_per_r > mean_before_zero, time_values > 0)
                    alpha_eff_per_nue = np.mean(alpha_vals[:, time_with_enough_acc], axis=1)
                    if alpha_eff_per_nue_all is None:
                        alpha_eff_per_nue_all = alpha_eff_per_nue
                    else:
                        alpha_eff_per_nue_all = np.vstack((alpha_eff_per_nue_all, alpha_eff_per_nue))
                    thresholds_list.append(mean_before_zero)
                    alpha_vals_list.append(alpha_vals)
                    num_dates += 1

    # Visual results
    plt.figure(figsize=(10, 10))
    sns.heatmap(np.transpose(alpha_eff_per_nue_all), annot=False, cmap="YlGnBu", cbar=True)
    plt.title('Changes in Neurons States Across Dates')
    plt.ylabel('Neurons States')
    plt.xlabel('Dates')
    plt.show()
    plt.savefig(os.path.join(directory, 'neurons_states_across_dates.png'))

    # Find in/active neurons
    max_th = 0.9
    min_th = 0.2
    active_neurons_all_dates = np.all(alpha_eff_per_nue_all > max_th, axis=0)
    active_neurons_all_dates = np.where(active_neurons_all_dates)[0]
    inactive_neurons_all_dates = np.all(alpha_eff_per_nue_all < min_th, axis=0)
    inactive_neurons_all_dates = np.where(inactive_neurons_all_dates)[0]

    # Change in order for better visualization
    all_indices = set(range(alpha_vals.shape[0]))
    active_set = set(active_neurons_all_dates)
    inactive_set = set(inactive_neurons_all_dates)
    remaining_indices = all_indices - active_set - inactive_set
    organized_indices = np.array(list(active_set) + list(remaining_indices) + list(inactive_set))
    plt.figure(figsize=(10, 10))
    sns.heatmap(np.transpose(alpha_eff_per_nue_all)[organized_indices, :], annot=False, cmap="YlGnBu", cbar=True)
    plt.title('Changes in Neurons States Across Dates')
    plt.ylabel('Neurons States')
    plt.xlabel('Dates')
    plt.show()
    plt.savefig(os.path.join(directory, 'neurons_states_across_dates.png'))

    # Calculate the correlation matrix between of the dates for each of the neurons
    stacked_alpha_vals = np.stack(alpha_vals_list)
    correlation_matrices = np.zeros((alpha_vals.shape[0], num_dates, num_dates))
    for neuron in range(alpha_vals.shape[0]):
        neuron_data = stacked_alpha_vals[:, neuron, :]
        correlation_matrices[neuron] = np.corrcoef(neuron_data)  # Pearson correlation coefficient
    # Create the heatmap of 1 neuron
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrices[active_neurons_all_dates[0]],
                annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('Correlation Matrix across dates experiments for 1 neuron')
    plt.xlabel('Dates')
    plt.ylabel('Dates')

    # Create the heatmap of active_neurons_all_dates mean
    plt.figure(figsize=(8, 6))
    sns.heatmap(np.mean(correlation_matrices[active_neurons_all_dates], axis=0),
                annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('Correlation Matrix across dates experiments for 1 neuron')
    plt.xlabel('Dates')
    plt.ylabel('Dates')

    scipy.io.savemat(os.path.join(directory, 'find_indicative_neurons.mat'),
                     {'alpha_eff_per_nue_all': alpha_eff_per_nue_all,
                      'thresholds_list': thresholds_list,
                      'active_neurons_all_dates': active_neurons_all_dates,
                      'inactive_neurons_all_dates': inactive_neurons_all_dates
                      })


def find_dominant_neurons(matrices):
    # Set the thresholds
    min_th = 0.2
    max_th = 0.8

    # List to store row indices of changing neurons in each matrix
    changing_neurons_per_matrix = []

    # Iterate over each matrix
    for matrix in matrices:
        # Find row indices of neurons that change from below min_th to above max_th
        changing_neurons_rows = np.where((matrix[:, 0] <= min_th) & (matrix[:, -1] >= max_th))[0]
        changing_neurons_per_matrix.append(changing_neurons_rows)

    # Find row indices of neurons that change in all matrices
    common_changing_neurons = set(changing_neurons_per_matrix[0])
    for changing_neurons in changing_neurons_per_matrix[1:]:
        common_changing_neurons.intersection_update(set(changing_neurons))

    # Convert the set to a list for easier use
    common_changing_neurons_list = list(common_changing_neurons)

    # Display results
    print("Changing neurons per matrix:")
    for i, changing_neurons in enumerate(changing_neurons_per_matrix):
        print(f"Matrix {i + 1}: {changing_neurons}")

    print("\nCommon changing neurons across all matrices:")
    print(common_changing_neurons_list)


directory = '/home/shiralif/results/4575_fixLoss'
thresholds = find_indicative_neurons(directory)
print(thresholds)
