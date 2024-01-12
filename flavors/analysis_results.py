import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pandas as pd
import matplotlib.gridspec as gridspec
from c_stg.params import Params
from data_processing import DataProcessor
from data_params import data_origanization_params

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
        pattern = rf"{key} = ([\w._-]+)"
        matches = re.findall(pattern, content)

        if len(matches) >= occurrence:
            return matches[occurrence - 1]
        else:
            return None


def find_indicative_neurons(directory, excel_df, context, sub_title=""):
    #
    # date2idx = {'05.03.19': 0, '10.03.19': 1, '14.03.19': 2, '19.03.19': 3, '31.03.19': 4, '03.04.19': 5, '07.04.19': 6,
    #             '11.04.19': 7, '15.04.19': 8}
    date2idx = {'14.03.19': 0, '19.03.19': 1, '31.03.19': 2, '03.04.19': 3, '07.04.19': 4,
                '11.04.19': 5, '15.04.19': 6}
    num_dates = len(date2idx)

    alpha_eff_per_nue_all = np.zeros((num_dates, 306))
    thresholds_list = [None]*num_dates
    date_list = [None]*num_dates
    flavors_list = [None]*num_dates
    type1_list = [None]*num_dates
    type2_list = [None]*num_dates

    all_acc_vals_per_r = np.full((num_dates, 4), None, dtype=object)
    all_alpha_vals = []

    # Create a figure for flavors correlation figure for flavors context
    fig = plt.figure(figsize=(20, 10))  # Width for 4 plots, height for 2 plots
    gs = gridspec.GridSpec(2, 4, figure=fig)
    cbar_ax = fig.add_subplot(gs[0, 3])


    vmin, vmax = 1, -1
    ax_idx = 0
    for subdir in os.listdir(directory):
        if not subdir.endswith('.mat') and not subdir.endswith('.png'):
            for file in os.listdir(os.path.join(directory, subdir)):
                if file.endswith('.mat'):
                    mat_path = os.path.join(os.path.join(directory, subdir), file)
                    date = mat_path.split('\\')[-2].split('_')[-4:-1]
                    date = [date[1], date[0], date[2]]
                    date = '.'.join(date)

                    folder = '_'.join(mat_path.split('\\')[-2].split('_')[-4:-1])
                    type1 = str(excel_df[excel_df['folder'] == folder]['type1'].iloc[0])
                    flavors = str(excel_df[excel_df['folder'] == folder]['flavors'].iloc[0]).split('_')
                    type2 = str(excel_df[excel_df['folder'] == folder]['type2'].iloc[0])

                    data = scipy.io.loadmat(mat_path)
                    acc_vals_per_r = data['acc_vals_per_r'][0]
                    alpha_vals = data['alpha_vals']

                    log_file = os.path.join(os.path.join(directory, subdir), "log.txt")
                    start_time = float(extract_value(log_file, "start_time", occurrence=1))
                    end_time = float(extract_value(log_file, "end_time", occurrence=2))
                    chance_level = float(extract_value(log_file, "chance_level", occurrence=1))

                    if context == 'time:':
                        time_values = np.linspace(start_time, end_time, acc_vals_per_r.shape[0])
                        time_with_enough_acc = np.logical_and(acc_vals_per_r > chance_level, time_values > 0)
                        #time_with_enough_acc = np.logical_and(acc_vals_per_r > chance_level, time_values <= 0)
                        #time_with_enough_acc = np.logical_and(acc_vals_per_r > chance_level, time_values > 2, time_values < 4)
                        #time_with_enough_acc = time_values <= 0
                        alpha_eff_per_nue = np.mean(alpha_vals[:, time_with_enough_acc], axis=1)

                        alpha_eff_per_nue_all[date2idx[date], :] = alpha_eff_per_nue

                    elif context == 'flavors':
                        if acc_vals_per_r.shape[0] == 3:
                            all_acc_vals_per_r[date2idx[date], :-1] = acc_vals_per_r
                        elif acc_vals_per_r.shape[0] == 4:
                            all_acc_vals_per_r[date2idx[date], :] = acc_vals_per_r

                        flavors_corr_mat = np.corrcoef(np.transpose(alpha_vals))  # Pearson correlation coefficient
                        #plt.figure(figsize=(10, 10))
                        vmin, vmax = min(vmin, flavors_corr_mat.min()), max(vmax, flavors_corr_mat.max())

                        # ax = sns.heatmap(flavors_corr_mat,
                        #                  annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
                        if ax_idx == 3:
                            ax_idx += 1

                        ax = plt.subplot(gs[ax_idx])
                        ax_idx += 1

                        sns.heatmap(flavors_corr_mat, annot=True, cmap='coolwarm', square=True, ax=ax, cbar=False)
                        ax.set_title(f'{date}, {type1}, {type2}')
                        ax.set_xticklabels(flavors, rotation=0)
                        ax.set_yticklabels(flavors, rotation=0)

                    all_alpha_vals.append(alpha_vals)

                    thresholds_list[date2idx[date]] = chance_level
                    date_list[date2idx[date]] = date
                    flavors_list[date2idx[date]] = flavors
                    type1_list[date2idx[date]] = type1
                    type2_list[date2idx[date]] = type2

    if context == 'flavors':
        for ax in fig.get_axes():
            for im in ax.collections:
                im.set_clim(vmin, vmax)

        fig.colorbar(fig.get_axes()[1].collections[0], cax=cbar_ax)
        plt.tight_layout()
        plt.savefig(os.path.join(directory, 'Corr_matrices_vs_flavors.png'))



    # Visual results
    combined_labels = [f"{date}\n{type1}\n{flavors}\n{type2}"
                       for date, type1, flavors, type2 in
                       zip(date_list, type1_list, flavors_list, type2_list)]

    if context == 'time':
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(np.transpose(alpha_eff_per_nue_all), annot=False, cmap="YlGnBu", cbar=True)
        plt.title('Changes in Neurons States Across Dates'+'\n'+sub_title)
        plt.ylabel('Neurons States')
        ax.set_xticklabels(combined_labels, rotation=0)
        #plt.xlabel('Dates')
        plt.savefig(os.path.join(directory, sub_title + 'neurons_states_across_dates.png'))

        # Find in/active neurons
        max_th = 0.9
        min_th = 0.2
        active_neurons_all_dates = np.all(alpha_eff_per_nue_all > max_th, axis=0)
        active_neurons_all_dates = np.where(active_neurons_all_dates)[0]
        print("active_neurons_all_dates:")
        print(active_neurons_all_dates)
        inactive_neurons_all_dates = np.all(alpha_eff_per_nue_all < min_th, axis=0)
        inactive_neurons_all_dates = np.where(inactive_neurons_all_dates)[0]
        print("inactive_neurons_all_dates")
        print(inactive_neurons_all_dates)

        # Change in order for better visualization
        ic = np.argsort(np.transpose(alpha_eff_per_nue_all)[:, -1])
        sorted_alpha_eff_per_nue = np.transpose(alpha_eff_per_nue_all)[np.flip(ic), :]
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(sorted_alpha_eff_per_nue, annot=False, cmap="YlGnBu", cbar=True)
        plt.title('Changes in Neurons States Across Dates'+'\n'+sub_title)
        plt.ylabel('Neurons States')
        ax.set_xticklabels(combined_labels, rotation=0)
        #plt.xlabel('Dates')
        plt.savefig(os.path.join(directory, sub_title+'neurons_states_across_dates_organized.png'))


        # Calculate the correlation matrix between each of the dates
        correlation_matrix = np.corrcoef(alpha_eff_per_nue_all)  # Pearson correlation coefficient
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(correlation_matrix,
                    annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
        plt.title('Correlation Matrix across dates experiments'+'\n'+sub_title)
        #plt.xlabel('Dates')
        ax.set_xticklabels(combined_labels, rotation=0)
        #plt.ylabel('Dates')
        ax.set_yticklabels(combined_labels, rotation=0)
        plt.savefig(os.path.join(directory, sub_title + 'Dates_correlation.png'))

    elif context == "flavors":

        # Flavors accuracy vs dates
        flavors = ['s', 'q', 'g', 'f']
        plt.figure(figsize=(10, 6))
        for i, flavor in enumerate(flavors):
            plt.plot(all_acc_vals_per_r[:, i], label=flavor, marker='o', linestyle='--')

        plt.xticks(ticks=np.arange(len(combined_labels)), labels=combined_labels, rotation=0)
        plt.legend()
        plt.title('Flavors accuracy vs dates ')
        plt.xlabel('Date of experiment')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join(directory, sub_title + 'Flavors_acc_vs_dates.png'))


        # Correlation between dates for each of the flavors separately
        for idx_flavor, flavor in enumerate(flavors):
            neu_vs_date_mat = None
            dates_inds_list = []
            for idx_date in range(len(all_alpha_vals)):
                if idx_flavor + 1 > all_alpha_vals[idx_date].shape[1]:
                    continue
                else:
                    if neu_vs_date_mat is None:
                        neu_vs_date_mat = all_alpha_vals[idx_date][:, idx_flavor]
                    else:
                        neu_vs_date_mat = np.vstack((neu_vs_date_mat, all_alpha_vals[idx_date][:, idx_flavor]))
                    dates_inds_list.append(idx_date)

            dates_corr_mat = np.corrcoef(neu_vs_date_mat)  # Pearson correlation coefficient
            plt.figure(figsize=(10, 10))
            ax = sns.heatmap(dates_corr_mat,
                        annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
            plt.title(f'Correlation Matrix across dates for flavor {flavor}')
            ax.set_xticklabels([combined_labels[i] for i in dates_inds_list], rotation=0)
            ax.set_yticklabels([combined_labels[i] for i in dates_inds_list], rotation=0)
            plt.savefig(os.path.join(directory, f'Corr_mat_vs_date_for_{flavor}.png'))


def open_gates_visual(directory, date, min_th=0.2, max_th=0.8, num_neu=5):

    params = Params()
    params.date = date
    params = data_origanization_params(params)
    assert params.context_key == 'time', "this function only suitable for time context option"

    # Write running parameters to a text file
    os.makedirs(params.res_directory, exist_ok=True)
    print('Writing results to %s\n' % params.res_directory)
    with open(os.path.join(params.res_directory, 'log.txt'), 'w') as f:
        f.write(''.join(["%s = %s\n" % (k, v) for k, v in params.__dict__.items()]))

    data = DataProcessor(params)
    end_time = float(extract_value(os.path.join(params.res_directory, 'log.txt'), "end_time", occurrence=2))
    start_time = float(extract_value(os.path.join(params.res_directory, 'log.txt'), "start_time", occurrence=1))
    time = np.linspace(start_time, end_time, np.unique(data.context_feat).shape[0])

    for file in os.listdir(directory):
        if file.endswith('.mat'):
            mat_path = os.path.join(directory, file)
            mat_data = scipy.io.loadmat(mat_path)
            acc_vals_per_r = mat_data['acc_vals_per_r'][0]
            alpha_vals = mat_data['alpha_vals']

            begin_low = alpha_vals[:, 0] < min_th
            end_high = alpha_vals[:, -1] > max_th
            opened_neuron_indices = np.where(begin_low & end_high)[0]

    fig = plt.figure(figsize=(10, 20))  # Width for 4 plots, height for 2 plots
    gs = gridspec.GridSpec(num_neu, 2+1, width_ratios=[1, 1, 0.05], figure=fig)
    # acc_vals_per_r
    ax = plt.subplot(gs[0])
    ax.plot(time, acc_vals_per_r)
    plt.xlabel("Time [sec]")
    plt.ylabel("Accuracy")
    ax.axvline(x=0, color='red', linestyle='--')  # Adding vertical line at 0 for Alpha Vals

    # all alpha_vals
    ax = plt.subplot(gs[1])
    ic = np.argsort(alpha_vals[:, 0])
    sorted_alpha_vals = alpha_vals[ic, :]
    cax = ax.imshow(sorted_alpha_vals, aspect='auto', extent=[time[0], time[-1], 0, alpha_vals.shape[0]])
    ax.axvline(x=0, color='red', linestyle='--')  # Adding vertical line at 0 for Alpha Vals
    plt.xlabel("Time [sec]")
    plt.ylabel("#neuron")
    ax_colorbar = plt.subplot(gs[2])
    plt.colorbar(cax, cax=ax_colorbar)

    ax_idx = 3
    sub_idx = 0
    chosen =[opened_neuron_indices[0], opened_neuron_indices[7], opened_neuron_indices[10], opened_neuron_indices[22]]
    for i in range(num_neu-1):
        if sub_idx % 2 == 0 and sub_idx != 0:
            sub_idx = 0
            ax_idx += 1
        neu_idx = chosen[i]#opened_neuron_indices[i+20]

        # alphas vs time per neuron
        ax = plt.subplot(gs[ax_idx])
        ax_idx += 1
        sub_idx += 1
        ax.plot(time, alpha_vals[neu_idx, :])
        ax.axvline(x=0, color='red', linestyle='--')
        plt.xlabel("Time [sec]")
        plt.ylabel(f"Cell weight {i}")

        # mean activity
        ax = plt.subplot(gs[ax_idx])
        ax_idx += 1
        sub_idx += 1
        activities, errors = data.get_neu_activity(neu_idx)
        labels = ["success trials", "failure trials"]
        colors = ["green", "orange"]
        for neu_activity, error, lab, color in zip(activities, errors, labels, colors):
            if ax_idx == 5:
                ax.plot(time, neu_activity, label=lab, color=color)
                plt.legend()
            else:
                ax.plot(time, neu_activity, color=color)
            ax.axvline(x=0, color='red', linestyle='--')
            ax.fill_between(time, neu_activity - error/2, neu_activity + error/2, alpha=0.5, color=color)# edgecolor='#CC4F1B', facecolor='#FF9848')

        plt.xlabel("Time [sec]")
        plt.ylabel("Meas")

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(directory, 'Cell gate activity examples.png'))



excel_path = '../data/animals_db_selected.xlsx'
excel_df = pd.read_excel(excel_path, sheet_name=1)
directory = '..\\results\\2024_01_01_20_48_09_animal_4575_date_03_19_19_success'
date = '03_19_19'
#sub_title = 'after tone with enough acc '
#sub_title = 'before tone with enough acc '
#sub_title = 'after tone, 2-4 sec, with enough acc '
#sub_title = 'before tone with no acc th'
#thresholds = find_indicative_neurons(directory, excel_df, sub_title)
# ba = find_indicative_neurons(directory, excel_df, context="flavors")
open_gates_visual(directory, date)
print("end")
