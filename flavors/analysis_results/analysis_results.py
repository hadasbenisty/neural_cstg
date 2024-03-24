import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pandas as pd
import matplotlib.gridspec as gridspec
from data_processing import DataProcessor
from data_params import data_origanization_params
from c_stg.params import Params_config
import scipy.io as spio
import ast
from flavors.utils import extract_value, find_best_hyper_comb


def find_indicative_neurons(directory, animal, context, sub_title=""):
    #
    #num2flavors = {1: 'g', 2: 's', 3: 'q',4: 'r',5: 'f',0: 'fail'}
    # num2flavors = {1: 'g', 2: 's', 3: 'q', 4: 'f', 0: 'fail'}  # Todo
    animal2sheet_num = {'4458': 0, '4575': 1, '4754': 2,
                        '4756': 3, '4880': 4, '4882': 5, '4940': 6, '1111': 7}  # excel sheet num
    excel_df = pd.read_excel('../data/animals_db_selected.xlsx', sheet_name=animal2sheet_num[animal])
    date2idx = init_date2idx(animal, context)

    num_dates = len(date2idx)

    mu_eff_per_nue_all = None #np.zeros((num_dates, 368))
    thresholds_list = [None]*num_dates
    date_list = [None]*num_dates
    flavors_list = [None]*num_dates
    type1_list = [None]*num_dates
    type2_list = [None]*num_dates
    flavors2num_dict_list = [None]*num_dates

    all_acc_vals_per_r = np.full((num_dates, 4), None, dtype=object)
    all_mu_vals = [None]*num_dates

    # Create a figure for flavors correlation figure for flavors context
    fig = plt.figure(figsize=(20, 10))  # Width for 4 plots, height for 2 plots
    gs = gridspec.GridSpec(2, 5, figure=fig)
    cbar_ax = fig.add_subplot(gs[0, 3])

    vmin, vmax = 1, -1
    ax_idx = 0
    for subdir in os.listdir(directory):
        if not subdir.endswith('.mat') and not subdir.endswith('.png') and not subdir =='old_results':
            for file in os.listdir(os.path.join(directory, subdir)):
                if file.endswith('.mat'):
                    mat_path = os.path.join(os.path.join(directory, subdir), file)
                    date = mat_path.split('\\')[-2].split('_')[-4:-1]
                    date = [date[1], date[0], date[2]]
                    date = '.'.join(date)

                    folder = '_'.join(mat_path.split('\\')[-2].split('_')[-4:-1])
                    type1 = str(excel_df[excel_df['folder'] == folder]['type1'].iloc[0])
                    type2 = str(excel_df[excel_df['folder'] == folder]['type2'].iloc[0])
                    flavors = str(excel_df[excel_df['folder'] == folder]['flavors'].iloc[0]).split('_')

                    data = scipy.io.loadmat(mat_path)
                    acc_vals_per_r = data['acc_vals_per_r'][0]
                    if animal == '4575' and context == 'time':
                        mu_vals = data['alpha_vals']
                    else:
                        mu_vals = data['mu_vals']

                    log_file = os.path.join(os.path.join(directory, subdir), "log.txt")
                    start_time = float(extract_value(log_file, "start_time", occurrence=1))
                    end_time = float(extract_value(log_file, "end_time", occurrence=2))
                    chance_level = float(extract_value(log_file, "chance_level", occurrence=1))
                    flavors2num = extract_value(log_file, "flavors2num", occurrence=1)
                    if animal == '4575' and context == 'time':
                        flavors2num_dict = None
                    elif animal == '4458' and context == 'time':
                        flavors2num_dict = None
                    else:
                        flavors2num_dict = ast.literal_eval(flavors2num)
                        # Sort flavors based on the values in the flavors2num dictionary
                        flavors = sorted(flavors, key=lambda flavor: flavors2num_dict[flavor])

                    # Replace 'r' with 'g' in a list of lists - relevant for animal 4458 only
                    for i, element in enumerate(flavors):
                        if element == 'r':
                            flavors[i] = 'g'

                    if context == 'time':
                        time_values = np.linspace(start_time, end_time, acc_vals_per_r.shape[0])
                        time_with_enough_acc = np.logical_and(acc_vals_per_r > chance_level, time_values > 0)
                        #time_with_enough_acc = np.logical_and(acc_vals_per_r > chance_level, time_values <= 0)
                        #time_with_enough_acc = np.logical_and(acc_vals_per_r > chance_level, time_values > 2, time_values < 4)
                        #time_with_enough_acc = time_values <= 0
                        mu_eff_per_nue = np.mean(mu_vals[:, time_with_enough_acc], axis=1)
                        if mu_eff_per_nue_all is None:
                            mu_eff_per_nue_all = np.zeros((num_dates, mu_eff_per_nue.shape[0]))
                            mu_eff_per_nue_all[date2idx[date], :] = mu_eff_per_nue
                        else:
                            mu_eff_per_nue_all[date2idx[date], :] = mu_eff_per_nue

                    # Calculate the correlation matrix between each of the flavors per dates
                    elif context == 'flavors':
                        if acc_vals_per_r.shape[0] == 3:
                            all_acc_vals_per_r[date2idx[date], :-1] = acc_vals_per_r
                        elif acc_vals_per_r.shape[0] == 4:
                            all_acc_vals_per_r[date2idx[date], :] = acc_vals_per_r

                        flavors_corr_mat = np.corrcoef(np.transpose(mu_vals))  # Pearson correlation coefficient
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
                        ax.set_xticklabels(flavors, rotation=0, fontsize=12)
                        ax.set_yticklabels(flavors, rotation=0, fontsize=12)


                    #all_mu_vals.append(mu_vals)
                    all_mu_vals[date2idx[date]] = mu_vals
                    thresholds_list[date2idx[date]] = chance_level
                    date_list[date2idx[date]] = date
                    flavors_list[date2idx[date]] = flavors
                    type1_list[date2idx[date]] = type1
                    type2_list[date2idx[date]] = type2
                    flavors2num_dict_list[date2idx[date]] = flavors2num_dict

    # # Replace 'r' with 'g' in a list of lists - relevant for animal 4458 only
    # for sublist in flavors_list:
    #     for i, element in enumerate(sublist):
    #         if element == 'r':
    #             sublist[i] = 'g'

    if context == 'flavors':
        for ax in fig.get_axes():
            for im in ax.collections:
                im.set_clim(vmin, vmax)

        fig.colorbar(fig.get_axes()[1].collections[0], cax=cbar_ax)
        plt.tight_layout()
        plt.savefig(os.path.join(directory, 'Corr_matrices_vs_flavors_right_flavor_order.png'))



    # Visual results
    combined_labels = [f"{date}\n{type1}\n{flavors}\n{type2}"
                       for date, type1, flavors, type2 in
                       zip(date_list, type1_list, flavors_list, type2_list)]

    if context == 'time':
        plt.figure(figsize=(15, 10))
        ax = sns.heatmap(np.transpose(mu_eff_per_nue_all), annot=False, cmap="YlGnBu", cbar=True)
        plt.title('Changes in Neurons States Across Dates'+'\n'+sub_title)
        plt.ylabel('Neurons States')
        ax.set_xticklabels(combined_labels, rotation=0)
        #plt.xlabel('Dates')
        plt.savefig(os.path.join(directory, sub_title + 'neurons_states_across_dates.png'))

        # Find in/active neurons
        max_th = 0.9
        min_th = 0.2
        active_neurons_all_dates = np.all(mu_eff_per_nue_all > max_th, axis=0)
        active_neurons_all_dates = np.where(active_neurons_all_dates)[0]
        print("active_neurons_all_dates:")
        print(active_neurons_all_dates)
        inactive_neurons_all_dates = np.all(mu_eff_per_nue_all < min_th, axis=0)
        inactive_neurons_all_dates = np.where(inactive_neurons_all_dates)[0]
        print("inactive_neurons_all_dates")
        print(inactive_neurons_all_dates)

        # Change in order for better visualization
        ic = np.argsort(np.transpose(mu_eff_per_nue_all)[:, -1])
        sorted_mu_eff_per_nue = np.transpose(mu_eff_per_nue_all)[np.flip(ic), :]
        plt.figure(figsize=(15, 10))
        ax = sns.heatmap(sorted_mu_eff_per_nue, annot=False, cmap="YlGnBu", cbar=True)
        plt.title('Changes in Neurons States Across Dates'+'\n'+sub_title)
        plt.ylabel('Neurons States')
        ax.set_xticklabels(combined_labels, rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(directory, sub_title+'neurons_states_across_dates_organized.png'))


        # Calculate the correlation matrix between each of the dates
        correlation_matrix = np.corrcoef(mu_eff_per_nue_all)  # Pearson correlation coefficient
        plt.figure(figsize=(15, 10))
        ax = sns.heatmap(correlation_matrix,
                    annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
        plt.title('Correlation Matrix across dates experiments'+'\n'+sub_title)
        ax.set_xticklabels(combined_labels, rotation=0)
        ax.set_yticklabels(combined_labels, rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(directory, sub_title + 'Dates_correlation.png'))
        spio.savemat(os.path.join(directory, 'Dates_correlation.mat'),
                     {'correlation_matrix': correlation_matrix,
                      'combined_labels': combined_labels})

        # Calculate the correlation matrix between each of the experiment part
        main_parts = ['train', 'first', 'ongoing_batch', 'ongoing_random']
        mu_eff_per_nue_all_parts = np.zeros((len(main_parts), mu_eff_per_nue_all.shape[1]))
        for part_idx, part in enumerate(main_parts):
            mu_eff_per_nue_part = []
            for date_idx in range(mu_eff_per_nue_all.shape[0]):
                type1 = combined_labels[date_idx].split("\n")[1]
                type2 = combined_labels[date_idx].split("\n")[3]
                if part == 'train' and type1 == part:
                    mu_eff_per_nue_part.append(mu_eff_per_nue_all[date_idx, :])
                elif part == 'first' and type1 == part:
                    mu_eff_per_nue_part.append(mu_eff_per_nue_all[date_idx, :])
                elif part == 'ongoing_batch':
                    if type1 == 'ongoing' and type2 == 'batch':
                        mu_eff_per_nue_part.append(mu_eff_per_nue_all[date_idx, :])
                elif part == 'ongoing_random':
                    if type1 == 'ongoing' and type2 == 'random':
                        mu_eff_per_nue_part.append(mu_eff_per_nue_all[date_idx, :])
            if len(mu_eff_per_nue_part)==0:
                continue
            stacked_mu_eff_per_nue_part = np.stack(mu_eff_per_nue_part, axis=0)
            mean_mu_eff_per_neu_part = np.mean(stacked_mu_eff_per_nue_part, axis=0)
            mu_eff_per_nue_all_parts[part_idx,:] = mean_mu_eff_per_neu_part

        correlation_matrix_parts = np.corrcoef(mu_eff_per_nue_all_parts)  # Pearson correlation coefficient
        plt.figure(figsize=(15, 10))
        ax = sns.heatmap(correlation_matrix_parts,
                    annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
        plt.title('Correlation Matrix across experiment parts')
        ax.set_xticklabels(main_parts, rotation=0)
        ax.set_yticklabels(main_parts, rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(directory, sub_title + 'Parts_correlation.png'))
        spio.savemat(os.path.join(directory, 'Parts_correlation.mat'),
                     {'correlation_matrix_parts': correlation_matrix_parts,
                      'main_parts': main_parts})

    elif context == "flavors":

        # Check if all flavors2num dictionaries are the same
        try:
            check_dicts_are_same(flavors2num_dict_list)
            print("All dictionaries are the same.")
        except ValueError as e:
            print(e)

        # Correlation between dates for each of the flavors separately
        for flavor_key in flavors2num_dict_list[0]:
            neu_vs_date_mat = None  # for each of the flavors
            dates_inds_list = []
            for idx_date in range(len(all_mu_vals)):
                if flavor_key in flavors_list[idx_date]:
                    flav_idx_in_date = flavors_list[idx_date].index(flavor_key)
                    if neu_vs_date_mat is None:
                        neu_vs_date_mat = all_mu_vals[idx_date][:, flav_idx_in_date]
                    else:
                        dates_inds_list.append(idx_date)

            if neu_vs_date_mat is None:
                continue
            dates_corr_mat = np.corrcoef(neu_vs_date_mat)  # Pearson correlation coefficient
            plt.figure(figsize=(10, 10))
            ax = sns.heatmap(dates_corr_mat,
                             annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
            plt.title(f'Correlation Matrix across dates for flavor {flavor_key}')
            relevent_combined_labels = [combined_labels[i] for i in dates_inds_list]
            ax.set_xticklabels(relevent_combined_labels, rotation=0)
            ax.set_yticklabels(relevent_combined_labels, rotation=0)
            plt.savefig(os.path.join(directory, f'Corr_mat_vs_date_for_{flavor_key}_right_flavor_order.png'))
            spio.savemat(os.path.join(directory, f'Corr_mat_vs_date_for_{flavor_key}.mat'),
                         {'correlation_matrix': dates_corr_mat,
                          'combined_labels': relevent_combined_labels})

            # Calculate the correlation matrix between each of the experiment part
            #main_parts = ['train', 'first', 'ongoing_batch', 'ongoing_random']
            main_parts = ['first', 'ongoing_batch', 'ongoing_random']
            mu_one_flav_all_parts = np.zeros((len(main_parts), neu_vs_date_mat.shape[1]))
            for part_idx, part in enumerate(main_parts):
                mu_one_flav_nue_part = []
                for date_idx in range(neu_vs_date_mat.shape[0]):
                    type1 = relevent_combined_labels[date_idx].split("\n")[1]
                    type2 = relevent_combined_labels[date_idx].split("\n")[3]
                    if part == 'train' and type1 == part:
                        mu_one_flav_nue_part.append(neu_vs_date_mat[date_idx, :])
                    elif part == 'first' and type1 == part:
                        mu_one_flav_nue_part.append(neu_vs_date_mat[date_idx, :])
                    elif part == 'ongoing_batch':
                        if type1 == 'ongoing' and type2 == 'batch':
                            mu_one_flav_nue_part.append(neu_vs_date_mat[date_idx, :])
                    elif part == 'ongoing_random':
                        if type1 == 'ongoing' and type2 == 'random':
                            mu_one_flav_nue_part.append(neu_vs_date_mat[date_idx, :])
                if len(mu_one_flav_nue_part) == 0:
                    continue
                stacked_mu_eff_per_nue_part = np.stack(mu_one_flav_nue_part, axis=0)
                mean_mu_eff_per_neu_part = np.mean(stacked_mu_eff_per_nue_part, axis=0)
                mu_one_flav_all_parts[part_idx, :] = mean_mu_eff_per_neu_part

            corr_mat_parts_1_flav = np.corrcoef(mu_one_flav_all_parts)  # Pearson correlation coefficient
            plt.figure(figsize=(15, 10))
            ax = sns.heatmap(corr_mat_parts_1_flav,
                             annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
            plt.title(f'Correlation across experiment parts - {flavor_key} flavor')
            ax.set_xticklabels(main_parts, rotation=0)
            ax.set_yticklabels(main_parts, rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(directory, sub_title + f'Parts_correlation_{flavor_key}.png'))
            spio.savemat(os.path.join(directory, f'Parts_correlation_{flavor_key}.mat'),
                         {'correlation_matrix_parts': corr_mat_parts_1_flav,
                          'main_parts': main_parts})

def corr_parts_all_animals(context='_'):
    #  1 correlation matrix between parts - for all animals
    flavors = ['g', 's', 'q']

    animals = ['4458_0', '4575_1', '4754_2', '4756_3', '4880_4', '4882_5']
    if context == 'time':
        main_parts = ['train', 'first', 'ongoing_batch', 'ongoing_random']
        corr_parts_matrices = []
        for animal in animals:
            res_directory = '../results'
            animal_directory = os.path.join(res_directory, animal)
            animal_directory = os.path.join(animal_directory, 'time_context')
            corr_parts_animal_mat = scipy.io.loadmat(os.path.join(animal_directory, 'Parts_correlation.mat'))
            corr_parts_animal_mat = corr_parts_animal_mat['correlation_matrix_parts']
            corr_parts_matrices.append(corr_parts_animal_mat)
        # Stack the matrices and compute the mean
        stacked_matrices = np.stack(corr_parts_matrices, axis=0)
        # Compute the mean while ignoring NaN values
        corr_parts_total = np.nanmean(stacked_matrices, axis=0)
        # Calculate standard deviation while ignoring NaN values
        std_dev = np.nanstd(stacked_matrices, axis=0)
        # Count non-NaN elements along axis=0
        non_nan_count = np.sum(~np.isnan(stacked_matrices), axis=0)
        # Calculate the standard error
        standard_error = std_dev / np.sqrt(non_nan_count)
        # Plotting
        annotations = np.empty_like(corr_parts_total, dtype=object)
        for i in range(corr_parts_total.shape[0]):
            for j in range(corr_parts_total.shape[1]):
                    annotations[i, j] = f"{corr_parts_total[i, j]:.2f} ± {standard_error[i, j]:.2f}"

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(corr_parts_total, annot=annotations, fmt='', cmap='coolwarm', vmin=-1, vmax=1, square=True)
        plt.title(f'Correlation Matrix across experiment parts -  All animals')
        ax.set_xticklabels(main_parts, rotation=0)
        ax.set_yticklabels(main_parts, rotation=0)
        plt.savefig(os.path.join(res_directory, f'Corr_parts_all_animals_with_4458.png'))

    if context == 'flavors':
        main_parts = ['first', 'ongoing_batch', 'ongoing_random']
        for flavor_key in flavors:
            corr_parts_matrices_1_flav = []
            for animal in animals:
                res_directory = '../results'
                animal_directory = os.path.join(res_directory, animal)
                animal_directory = os.path.join(animal_directory, 'flavors_context')
                animal_directory = os.path.join(animal_directory, f'Parts_correlation_{flavor_key}.mat')
                if os.path.exists(animal_directory):
                    corr_parts_animal_mat = scipy.io.loadmat(animal_directory)
                else:
                    # File does not exist
                    continue
                corr_parts_animal_mat = corr_parts_animal_mat['correlation_matrix_parts']
                corr_parts_matrices_1_flav.append(corr_parts_animal_mat)
            # Stack the matrices and compute the mean
            stacked_matrices = np.stack(corr_parts_matrices_1_flav, axis=0)
            # Compute the mean while ignoring NaN values
            corr_parts_total = np.nanmean(stacked_matrices, axis=0)
            # Calculate standard deviation while ignoring NaN values
            std_dev = np.nanstd(stacked_matrices, axis=0)
            # Count non-NaN elements along axis=0
            non_nan_count = np.sum(~np.isnan(stacked_matrices), axis=0)
            print(flavor_key)
            print(non_nan_count)
            # Calculate the standard error
            standard_error = std_dev / np.sqrt(non_nan_count)
            # Plotting
            annotations = np.empty_like(corr_parts_total, dtype=object)
            for i in range(corr_parts_total.shape[0]):
                for j in range(corr_parts_total.shape[1]):
                    annotations[i, j] = f"{corr_parts_total[i, j]:.2f} ± {standard_error[i, j]:.2f}"

            plt.figure(figsize=(10, 10))
            ax = sns.heatmap(corr_parts_total, annot=annotations, fmt='', cmap='coolwarm', vmin=-1, vmax=1, square=True)
            plt.title(f'Correlation across experiment parts - All animals - {flavor_key} flavor')
            ax.set_xticklabels(main_parts, rotation=0)
            ax.set_yticklabels(main_parts, rotation=0)
            plt.savefig(os.path.join(res_directory, f'Corr_parts_all_animals_flavor_{flavor_key}.png'))


def check_dicts_are_same(dicts):
    if not dicts:  # If the list is empty, consider it as all "same"
        return
    first_dict = dicts[0]
    for i, d in enumerate(dicts[1:], start=1):  # Start comparing from the second item
        if d != first_dict:
            raise ValueError(f"Dictionaries are not the same at index 0 and {i}")

