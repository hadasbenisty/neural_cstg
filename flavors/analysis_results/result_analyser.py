# Imports
from utils import *
import scipy.io
from sklearn.metrics import confusion_matrix
import numpy as np
from flavors.data_processing import DataProcessor
from c_stg.params import Params_config
from flavors.data_params import data_origanization_params
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import defaultdict


class ResultsAnalyser:
    def __init__(self, res_directory, context, classification_type, animals, main_parts, animal2sheet_num,
                 info_excel_path, num_folds, flavors2num):
        self.res_main_directory = res_directory
        self.context = context
        self.classification_type = classification_type
        self.animals = animals
        self.main_parts = main_parts
        self.animal2sheet_num = animal2sheet_num
        self.flavors2num = flavors2num
        self.info_excel_path = info_excel_path
        self.num_folds = num_folds
        self.extract_run_type()
        self.curr_method = None

    def extract_run_type(self):

        if self.context == 'time' and self.classification_type == 'flavors':
            self.run_type = 'flavors_classification_new'
        elif self.context == 'time' and self.classification_type == 'success':
            self.run_type = 'time_context'
        elif self.context == 'flavors' and self.classification_type == 'success':
            self.run_type = 'flavors_context'
    def __load_data_property(self, data, chance_level, animal=None, date=None, time_idx=None, flavors=None):
        if self.curr_method == 'acc_vs_time_per_parts':
            unique_values = np.unique(data["acc_vals_per_r"][0])
            # Check if there is only one acc:
            if len(unique_values) == 1:
                print("Constant Accuracy")
                return None
            else:
                return data["acc_vals_per_r"][0] - chance_level
        elif self.curr_method == 'open_gate_percents_vs_time_per_parts':
            if "mu_vals" in data:
                open_percents_vs_t = (np.sum(data["mu_vals"] > 0.5, axis=0) / data["mu_vals"].shape[0])
            else:
                open_percents_vs_t = (np.sum(data["alpha_vals"] > 0.5, axis=0) / data["alpha_vals"].shape[0])
            return open_percents_vs_t
        elif self.curr_method == 'compare_percent_open_gates':
            if "mu_vals" in data:
                alpha_vals = data["mu_vals"], f'{animal}_{date}'
            else:
                alpha_vals = data["alpha_vals"], f'{animal}_{date}'
            return alpha_vals
        elif self.curr_method == 'conf_mat_at_specific_time_per_part':
            return data["targets_per_r"][time_idx][:], data["pred_labels_per_r"][time_idx][:], flavors, f'{animal}_{date}'
        elif self.curr_method == 'percent_open_gates_per_flavor_per_parts_vs_time_windows_' + 'open_gates':
            if "mu_vals" in data:
                return (np.sum(data["mu_vals"] > 0.5, axis=0) / data["mu_vals"].shape[0]), flavors
            else:
                return (np.sum(data["alpha_vals"] > 0.5, axis=0) / data["alpha_vals"].shape[0]), flavors

        elif self.curr_method == 'percent_open_gates_per_flavor_per_parts_vs_time_windows_' + 'acc_per_flavors':
            if "mu_vals" in data:
                return data["acc_vals_per_r"][0], flavors
            else:
                return data["acc_vals_per_r"][0], flavors
    def __load_results(self, include_chance_level, num_flavors_relevant=None, time_point=None):  # all animals

        parts_sublists = [[] for _ in range(len(self.main_parts))]

        for animal_num in self.animals:
            animal_directory = os.path.join(os.path.join(self.res_main_directory, animal_num), self.run_type)
            filtered_subdirs = [subdir for subdir in os.listdir(animal_directory) if self.__check_subdir_validity(subdir)]
            sorted_subdirs = sorted(filtered_subdirs, key=lambda x: extract_sortable_date(x))
            for subdir in sorted_subdirs:
                    print(subdir)
                    date_directory = os.path.join(animal_directory, subdir)
                    best_comb = find_best_hyper_comb(date_directory, key="nn_acc_dev")
                    comb_directory = os.path.join(date_directory, best_comb)

                    animal, date, type1, type2, chance_level, time_vals, flavors, time_idx=(
                        extract_from_log(date_directory, self.info_excel_path, include_chance_level,
                                        self.animal2sheet_num, self.flavors2num, time_point))

                    if num_flavors_relevant is not None:
                        if num_flavors_relevant != len(flavors):
                            continue

                    for idx_part, part in enumerate(self.main_parts):
                        if part == 'train' and type1 == part:
                            for i in range(self.num_folds):
                                # data = scipy.io.loadmat(os.path.join(comb_directory, f"selfold{i}.mat"))
                                # tmp = self.__load_data_property(data, chance_level, animal, date, time_idx, flavors)
                                # if tmp is not None:
                                #     parts_sublists[0].append(tmp)
                                AssertionError
                        elif part == 'first' and type1 == part:
                            for i in range(self.num_folds):
                                data = scipy.io.loadmat(os.path.join(comb_directory, f"selfold{i}.mat"))
                                tmp = self.__load_data_property(data, chance_level, animal, date, time_idx, flavors)
                                if tmp is not None:
                                    parts_sublists[0].append(tmp)
                        elif part == 'ongoing_batch':
                            if type1 == 'ongoing' and type2 == 'batch':
                                for i in range(self.num_folds):
                                    data = scipy.io.loadmat(os.path.join(comb_directory, f"selfold{i}.mat"))
                                    tmp = self.__load_data_property(data, chance_level, animal, date, time_idx, flavors)
                                    if tmp is not None:
                                        parts_sublists[1].append(tmp)
                        elif part == 'ongoing_random':
                            if type1 == 'ongoing' and type2 == 'random':
                                for i in range(self.num_folds):
                                    data = scipy.io.loadmat(os.path.join(comb_directory, f"selfold{i}.mat"))
                                    tmp = self.__load_data_property(data, chance_level, animal, date, time_idx, flavors)
                                    if tmp is not None:
                                        parts_sublists[2].append(tmp)
        return parts_sublists
    def __check_subdir_validity(self, subdir):
        if not subdir.endswith('txt') and not subdir.endswith('png') and not subdir.endswith(
                'mat') and not subdir == "old_results":
            return True
        else:
            return False

    def acc_vs_time_per_parts(self, num_flavors_relevant=None):   # time context, flavors/success classification
        self.curr_method = 'acc_vs_time_per_parts'
        if not self.context == 'time':
            raise ValueError("Not suitable context for this method")
        # for flavors classification the chnce level is not clearly determined
        include_chance_level = True if (self.classification_type == 'success' or
                                        self.classification_type == 'flavors' and num_flavors_relevant == 2)\
                               else False
        parts_sublists = (self.__load_results(include_chance_level=include_chance_level, num_flavors_relevant=num_flavors_relevant))

        # Plotting
        title_adding = "relative" if include_chance_level else ""
        title_adding_2 = str(num_flavors_relevant)+"_flav" if  self.classification_type == 'flavors' else ""
        save_path = os.path.join(self.res_main_directory, f'acc_{title_adding}_vs_time_parts_{self.run_type}_{title_adding_2}.png')
        plot_aligned_data_statistics(self.main_parts, parts_sublists,
                                     f"Mean {title_adding} accuracy vs time all animals - {self.run_type} ,{title_adding_2}",
                                     save_path, y_label="Mean accuracy", combined_plot=True)

    def open_gate_percents_vs_time_per_parts(self):  # time context, flavors/success classification
        self.curr_method = 'open_gate_percents_vs_time_per_parts'
        if not self.context == 'time':
            raise ValueError("Not suitable context for this method")
        parts_sublists = self.__load_results(include_chance_level=False)

        # Plotting
        save_path = os.path.join(self.res_main_directory, f'open_percent_vs_time_parts_{self.run_type}.png')
        plot_aligned_data_statistics(self.main_parts, parts_sublists, f"Open gates percentage vs time all animals - {self.run_type}",
                                     save_path, y_label="Open gates percentage", combined_plot=True)

    def compare_percent_open_gates(self):  # time context, flavors/success classification
        self.curr_method = 'compare_percent_open_gates'
        if not self.context == 'time':
            raise ValueError("Not suitable context for this method")
        self.classification_type = 'success'  # time_context
        self.extract_run_type()
        alpha_vals_parts_success_class = self.__load_results(include_chance_level=False)
        self.classification_type = 'flavors'  # flavors_classification_new
        self.extract_run_type()
        alpha_vals_parts_flavors_class = self.__load_results(include_chance_level=False)

        compare_percent_per_part = []
        for i, (parts_sucess_class, parts_flavors_class) in enumerate(
                zip(alpha_vals_parts_success_class, alpha_vals_parts_flavors_class)):
            curr_part_comparing = [[] for _ in range(len(self.main_parts))]
            for alpha_vals_success, alpha_vals_flavors in zip(parts_sucess_class, parts_flavors_class):
                if alpha_vals_success[1] != alpha_vals_flavors[1]:
                    raise ValueError("Not the same order!")
                else:
                    only_success, only_flavors, both = compare_matrices_columns(alpha_vals_success[0],
                                                                                alpha_vals_flavors[0])
                    curr_part_comparing[0].append(only_success)
                    curr_part_comparing[1].append(only_flavors)
                    curr_part_comparing[2].append(both)
                    # Todo: add calculation of the mean and the stds

            # Todo: maybe create a seperate function n utils for this plotting part
            options = ["only_success_open", "only_flavors_open", "both_open"]
            save_path = os.path.join(self.res_main_directory, f'Comparing percentage of open gates - part {self.main_parts[i]}.png')
            plot_aligned_data_statistics(options, curr_part_comparing,
                                         f'Comparing percentage of open gates - part {self.main_parts[i]}',
                                         save_path, combined_plot=True)

            compare_percent_per_part.append(curr_part_comparing)

    def conf_mat_at_specific_time_per_part(self, time_point):  # time context, flavors classification

        self.curr_method = 'conf_mat_at_specific_time_per_part'
        self.run_type = 'flavors_classification_new_with_targets'
        if not self.context == 'time' or not self.classification_type == 'flavors':
            raise ValueError("Not suitable context for this method")

        parts_sublists = self.__load_results(include_chance_level=False, time_point=time_point)
        confusion_parts_sublists = [[] for _ in range(len(parts_sublists))]

        for i, data_list in enumerate(parts_sublists):
            # Assuming 'data_list' is your list containing the sublists with tuples
            # Each tuple in the format: (targets, labels, flavors, animal_date)
            # Step 1: Group data by 'animal+date' and 'flavors'
            grouped_data = defaultdict(list)
            for targets, labels, flavors, animal_date in data_list:
                grouped_data[(animal_date, tuple(flavors))].append((targets, labels))

            # This will create a dictionary where each key is a unique combination of 'animal+date' and 'flavors',
            # and each value is a list of tuples containing 'targets' and 'labels'

            # Step 2 & 3: Concatenate labels and targets for each group and calculate confusion matrices
            confusion_matrices = []
            for (animal_date, flavors), group in grouped_data.items():
                # Concatenate all labels and targets in the group
                all_targets = np.concatenate([g[0] for g in group])
                all_labels = np.concatenate([g[1] for g in group])

                # Calculate the confusion matrix
                matrix = confusion_matrix(all_labels, all_targets.squeeze())
                # row_sums = matrix.sum(axis=1)[:, np.newaxis]  # Calculate the sum of each row
                # safe_row_sums = np.where(row_sums != 0, row_sums, 1)  # Replace zeros with ones to avoid division by zero
                # matrix = matrix.astype('float') / safe_row_sums
                matrix = matrix.astype('float')/matrix.sum()
                print(matrix)

                # Store the confusion matrix with its corresponding animal_date and flavors
                confusion_parts_sublists[i].append((matrix, flavors))

        fig, axes = plt.subplots(nrows=1, ncols=len(confusion_parts_sublists), figsize=(20, 7))

        for idx, (title, mat_list) in enumerate(zip(self.main_parts, confusion_parts_sublists)):
            mean_conf, err_conf = combine_list_of_conf_mat(mat_list, self.flavors2num)
            cax = axes[idx].matshow(mean_conf, cmap=plt.cm.Blues)
            axes[idx].set_title(title)

            # Set axis labels
            labels = list(self.flavors2num.keys())
            axes[idx].set_xticks(np.arange(len(labels)))  # Set the positions for x ticks
            axes[idx].set_yticks(np.arange(len(labels)))
            axes[idx].set_xlabel('Predicted labels')
            axes[idx].set_ylabel('True labels')
            axes[idx].set_xticklabels(labels)
            axes[idx].set_yticklabels(labels)

            # Annotate each cell with the mean and std values
            for (i, j), val in np.ndenumerate(mean_conf):
                axes[idx].text(j, i, f'{mean_conf[i, j]:.2f}\n±{err_conf[i, j]:.2f}', ha='center', va='center',
                               color='black')

        plt.tight_layout()
        fig.colorbar(cax, ax=axes.ravel().tolist(), orientation='vertical')
        title = f"Confusion matrix for time {str(time_point)} parts"
        fig.suptitle(title, fontsize=16)
        plt.savefig(os.path.join(self.res_main_directory, title + ".png"))

    def open_gates_visual(self, directory, min_th=0.2, max_th=0.8, num_neu=5):  # time context
        # specific animal+date directory need to be given for example:
        # 2024_01_15_01_13_09_animal_4880_date_07_21_19_success
        if not self.context == 'time':
            raise ValueError("Not suitable context for this method")
        date = directory[-16:-8]
        animal = directory[-26:-22]
        mat_files_directory = '../../data/'  # 'D:\\flavorsProject\data\\'  # '/home/shiralif/data/'
        result_directory = '../../results/'  # '..\\results\\'  # '/home/shiralif/results/'
        info_excel_path = '../../data/animals_db_selected.xlsx'
        data_args = {'animal': animal, 'date': date, 'context_key': self.context,
                     'mat_files_directory': mat_files_directory, 'result_directory': result_directory, 'info_excel_path': info_excel_path,
                     'outcome_keys': [self.classification_type], 'context_key': self.context}
        params = Params_config('flavors', data_kwargs=data_args)
        params = data_origanization_params(params)
        full_animal = animal + '_' + str(self.animal2sheet_num[animal])
        self.extract_run_type()
        directory_path = os.path.join(os.path.join(os.path.join(self.res_main_directory, full_animal), self.run_type), directory)
        params.res_directory = directory_path
        params.post_process_mode = True

        data = DataProcessor(params)
        _, _, _, _, _, time_values, _, _ = extract_from_log(directory_path, info_excel_path, False,
                                                self.animal2sheet_num, self.flavors2num, None)
        log_file = os.path.join(directory_path, "log.txt")
        chance_level = float(extract_value(log_file, "chance_level", occurrence=1))

        best_comb = find_best_hyper_comb(directory_path, key="nn_acc_dev")
        comb_directory = os.path.join(directory_path, best_comb)

        acc_list = []
        alphas_list = []
        for i in range(self.num_folds):
            data_tmp = scipy.io.loadmat(os.path.join(comb_directory, f"selfold{i}.mat"))
            self.curr_method = 'acc_vs_time_per_parts'
            acc_list.append(self.__load_data_property(data_tmp, chance_level))
            self.curr_method = 'compare_percent_open_gates'
            alphas_list.append(self.__load_data_property(data_tmp, chance_level, animal=animal, date=date)[0])
            self.curr_method = 'open_gates_visual'

        mean_acc = np.mean(np.array(acc_list), axis=0)
        std_acc = np.std(np.array(acc_list), axis=0)
        err_acc = std_acc / np.sqrt(len(acc_list))

        fig = plt.figure(figsize=(15, 18))  # Width for 4 plots, height for 2 plots
        gs = gridspec.GridSpec(num_neu, 2 + 1, width_ratios=[1, 1, 0.05], figure=fig)
        # acc_vals_per_r
        ax = plt.subplot(gs[0])
        ax.fill_between(time_values, mean_acc - err_acc, mean_acc + err_acc, alpha=0.5,)
        ax.plot(time_values, mean_acc)
        ax.set_xlabel("Time [sec]", fontsize=14)
        ax.set_ylabel("Relative accuracy", fontsize=14)
        ax.axvline(x=0, color='red', linestyle='--')  # Adding vertical line at 0 for mu Vals
        ax.set_title("Accuracy per context", fontsize=14)

        # all mu_vals
        ax = plt.subplot(gs[1])
        mu_vals = alphas_list[0]
        ic = np.argsort(mu_vals[:, -1])  # randomly choose te first alphas matrix
        sorted_mu_vals = mu_vals[ic, :]
        cax = ax.imshow(sorted_mu_vals, aspect='auto', extent=[time_values[0], time_values[-1], 0, mu_vals.shape[0]])
        ax.axvline(x=0, color='red', linestyle='--')  # Adding vertical line at 0 for mu Vals
        ax.set_xlabel("Time [sec]", fontsize=14)
        ax.set_ylabel("#neuron", fontsize=14)
        ax.set_title("Stochastic gates values", fontsize=14)
        ax_colorbar = plt.subplot(gs[2])
        plt.colorbar(cax, cax=ax_colorbar)

        begin_low = mu_vals[:, 0] < min_th
        end_high = mu_vals[:, -1] > max_th
        opened_neuron_indices = np.where(begin_low & end_high)[0]

        ax_idx = 3
        sub_idx = 0
        # chosen = [opened_neuron_indices[0], opened_neuron_indices[7], opened_neuron_indices[10],
        #           opened_neuron_indices[22]]
        chosen = [opened_neuron_indices[1],
                  opened_neuron_indices[10],
                  opened_neuron_indices[19],
                  opened_neuron_indices[17]]
        for i in range(num_neu - 1):
            sub_idx, ax_idx = plot_1_open_gate(sub_idx, ax_idx, i, chosen, data, gs, time_values, mu_vals)
        plt.tight_layout()
        plt.savefig(os.path.join(self.res_main_directory, 'Cell gate activity examples.png'))

    def __correlation_mat_time_context_1_animal(self, animal):


        self.curr_method = 'correlation_mat_between_parts_time_context_1_animal'
        if not self.context == 'time':
            raise ValueError("Not suitable context for this method")

        animal_directory = os.path.join(os.path.join(self.res_main_directory,
                                                     animal+'_'+ str(self.animal2sheet_num[animal])), self.run_type)
        filtered_subdirs = [subdir for subdir in os.listdir(animal_directory) if self.__check_subdir_validity(subdir)]
        sorted_subdirs = sorted(filtered_subdirs, key=lambda x: extract_sortable_date(x))

        date_list = []
        flavors_list = []
        type1_list = []
        type2_list = []
        all_eff_mu_vals = []


        for subdir in sorted_subdirs:
            date_directory = os.path.join(animal_directory, subdir)
            best_comb = find_best_hyper_comb(date_directory, key="nn_acc_dev")
            comb_directory = os.path.join(date_directory, best_comb)
            animal, date, type1, type2, chance_level, time_vals, flavors, time_idx = (
                extract_from_log(date_directory, self.info_excel_path, True,
                             self.animal2sheet_num, self.flavors2num, None))

            mu_eff_per_neu_date = []
            num_pass = 0
            for i in range(self.num_folds):
                data = scipy.io.loadmat(os.path.join(comb_directory, f"selfold{i}.mat"))
                acc_vals_per_r = data['acc_vals_per_r'][0]
                #all_acc_vals_per_r.append(acc_vals_per_r)
                if "alpha_vals" in data:
                    mu_vals = data['alpha_vals']
                else:
                    mu_vals = data['mu_vals']

                time_with_enough_acc = np.logical_and(acc_vals_per_r > chance_level, time_vals > 0)
                if not np.any(time_with_enough_acc):
                    print("pass")
                    num_pass += 1
                    continue
                # time_with_enough_acc = np.logical_and(acc_vals_per_r > chance_level, time_values <= 0)
                # time_with_enough_acc = np.logical_and(acc_vals_per_r > chance_level, time_values > 2, time_values < 4)
                # time_with_enough_acc = time_values <= 0
                mu_eff_per_nue = np.mean(mu_vals[:, time_with_enough_acc], axis=1)
                mu_eff_per_neu_date.append(mu_eff_per_nue)
            if num_pass == self.num_folds:
                continue
            mu_vals = np.mean(np.array(mu_eff_per_neu_date), axis=0)
            # Calculate the correlation matrix between each of the flavors per dates
            all_eff_mu_vals.append(mu_vals)
            date_list.append(date)
            flavors_list.append(flavors)
            type1_list.append(type1)
            type2_list.append(type2)

        combined_labels = [f"{date}\n{type1}\n{flavors}\n{type2}"
                           for date, type1, flavors, type2 in
                           zip(date_list, type1_list, flavors_list, type2_list)]
        all_eff_mu_vals = np.stack(all_eff_mu_vals, axis=0)
        visual_correlation_1_animal(animal_directory, animal, all_eff_mu_vals, combined_labels)

        # Calculate the correlation matrix between each of the experiment part
        main_parts = ['train', 'first', 'ongoing_batch', 'ongoing_random']
        mu_eff_per_nue_all_parts = np.zeros((len(main_parts), all_eff_mu_vals.shape[1]))
        for part_idx, part in enumerate(main_parts):
            mu_eff_per_nue_part = []
            for date_idx in range(all_eff_mu_vals.shape[0]):
                type1 = combined_labels[date_idx].split("\n")[1]
                type2 = combined_labels[date_idx].split("\n")[3]
                if part == 'train' and type1 == part:
                    mu_eff_per_nue_part.append(all_eff_mu_vals[date_idx, :])
                elif part == 'first' and type1 == part:
                    mu_eff_per_nue_part.append(all_eff_mu_vals[date_idx, :])
                elif part == 'ongoing_batch':
                    if type1 == 'ongoing' and type2 == 'batch':
                        mu_eff_per_nue_part.append(all_eff_mu_vals[date_idx, :])
                elif part == 'ongoing_random':
                    if type1 == 'ongoing' and type2 == 'random':
                        mu_eff_per_nue_part.append(all_eff_mu_vals[date_idx, :])
            if len(mu_eff_per_nue_part) == 0:
                continue
            stacked_mu_eff_per_nue_part = np.stack(mu_eff_per_nue_part, axis=0)
            mean_mu_eff_per_neu_part = np.mean(stacked_mu_eff_per_nue_part, axis=0)
            mu_eff_per_nue_all_parts[part_idx, :] = mean_mu_eff_per_neu_part

        correlation_matrix_parts = np.corrcoef(mu_eff_per_nue_all_parts)  # Pearson correlation coefficient
        plt.figure(figsize=(15, 10))
        ax = sns.heatmap(correlation_matrix_parts,
                         annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
        plt.title('Correlation Matrix across experiment parts')
        ax.set_xticklabels(main_parts, rotation=0)
        ax.set_yticklabels(main_parts, rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(animal_directory, 'Parts_correlation.png'))
        spio.savemat(os.path.join(animal_directory, 'Parts_correlation.mat'),
                     {'correlation_matrix_parts': correlation_matrix_parts,
                      'main_parts': main_parts})

        return correlation_matrix_parts

    def corr_parts_time_context_all_animals(self):

        corr_parts_matrices = []
        main_parts = ['train', 'first', 'ongoing_batch', 'ongoing_random']

        for animal_num in self.animals:
            animal = animal_num[:-2]
            corr_mat_parts = self.__correlation_mat_time_context_1_animal(animal)
            corr_parts_matrices.append(corr_mat_parts)

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
        plt.savefig(os.path.join(self.res_main_directory, f'Corr_parts_all_animals.png'))

    def __correlation_mat_flavors_context_1_animal(self, animal):
        self.curr_method = '__correlation_mat_flavors_context_1_animal'
        if not self.context == 'flavors':
            raise ValueError("Not suitable context for this method")

        animal_directory = os.path.join(os.path.join(self.res_main_directory,
                                                     animal+'_'+ str(self.animal2sheet_num[animal])), self.run_type)
        filtered_subdirs = [subdir for subdir in os.listdir(animal_directory) if self.__check_subdir_validity(subdir)]
        sorted_subdirs = sorted(filtered_subdirs, key=lambda x: extract_sortable_date(x))

        date_list = []
        flavors_list = []
        type1_list = []
        type2_list = []

        #all_acc_vals_per_r = np.full((num_dates, 4), None, dtype=object)
        all_acc_vals_per_r = []
        all_mu_vals = []

        # Create a figure for flavors correlation figure for flavors context
        fig = plt.figure(figsize=(20, 10))  # Width for 4 plots, height for 2 plots
        gs = gridspec.GridSpec(2, 5, figure=fig)
        cbar_ax = fig.add_subplot(gs[0, 3])
        vmin, vmax = 1, -1
        ax_idx = 0

        main_parts = ['first', 'ongoing_batch', 'ongoing_random']
        flavors_corr_mat_list_parts = [[] for _ in range(len(main_parts))]

        for subdir in sorted_subdirs:
            date_directory = os.path.join(animal_directory, subdir)
            best_comb = find_best_hyper_comb(date_directory, key="nn_acc_dev")
            comb_directory = os.path.join(date_directory, best_comb)
            animal, date, type1, type2, chance_level, time_vals, flavors, time_idx = (
                extract_from_log(date_directory, self.info_excel_path, True,
                                 self.animal2sheet_num, self.flavors2num, None))

            date_mu_vals = []
            for i in range(self.num_folds):
                data = scipy.io.loadmat(os.path.join(comb_directory, f"selfold{i}.mat"))
                acc_vals_per_r = data['acc_vals_per_r'][0]
                if len(acc_vals_per_r) != len(flavors):
                    print("passs")
                    continue
                all_acc_vals_per_r.append(acc_vals_per_r)
                if "alpha_vals" in data:
                    date_mu_vals.append(data['alpha_vals'])
                else:
                    date_mu_vals.append(data['mu_vals'])

            mu_vals = np.mean(np.array(date_mu_vals), axis=0)
            all_mu_vals.append(mu_vals)

            flavors_corr_mat = np.corrcoef(np.transpose(mu_vals))  # Pearson correlation coefficient
            vmin, vmax = min(vmin, flavors_corr_mat.min()), max(vmax, flavors_corr_mat.max())

            if ax_idx == 3:
                ax_idx += 1

            ax = plt.subplot(gs[ax_idx])
            ax_idx += 1
            sns.heatmap(flavors_corr_mat, annot=True, cmap='coolwarm', square=True, ax=ax, cbar=False)
            ax.set_title(f'{date}, {type1}, {type2}')
            ax.set_xticklabels(flavors, rotation=0, fontsize=12)
            ax.set_yticklabels(flavors, rotation=0, fontsize=12)

            date_list.append(date)
            flavors_list.append(flavors)
            type1_list.append(type1)
            type2_list.append(type2)

            # add to parts of flavors correlation matrices
            for part_idx, part in enumerate(main_parts):
                if part == 'train' and type1 == part:
                    AssertionError
                elif part == 'first' and type1 == part:
                    flavors_corr_mat_list_parts[0].append((flavors_corr_mat, flavors))
                elif part == 'ongoing_batch':
                    if type1 == 'ongoing' and type2 == 'batch':
                        flavors_corr_mat_list_parts[1].append((flavors_corr_mat, flavors))
                elif part == 'ongoing_random':
                    if type1 == 'ongoing' and type2 == 'random':
                        flavors_corr_mat_list_parts[2].append((flavors_corr_mat, flavors))


        for ax in fig.get_axes():
            for im in ax.collections:
                im.set_clim(vmin, vmax)

        fig.colorbar(fig.get_axes()[1].collections[0], cax=cbar_ax)
        plt.tight_layout()
        plt.savefig(os.path.join(animal_directory, 'Corr_matrices_vs_flavors_all_dates.png'))

        # Visual results
        combined_labels = [f"{date}\n{type1}\n{flavors}\n{type2}"
                           for date, type1, flavors, type2 in
                           zip(date_list, type1_list, flavors_list, type2_list)]

        # Correlation between dates for each of the flavors separately
        corr_mats_parts_per_flavor = [[] for _ in range(len(self.flavors2num))]
        flavors2num_tmp = {'g': 1, 's': 2, 'q': 3}
        for idx_flavor, flavor_key in enumerate(flavors2num_tmp):
            neu_vs_date_mat = []  # for each of the flavors
            dates_inds_list = []
            for idx_date in range(len(all_mu_vals)):
                try:
                    flav_idx_in_date = flavors_list[idx_date].index(flavor_key)
                except ValueError:
                    continue
                neu_vs_date_mat.append(all_mu_vals[idx_date][:, flav_idx_in_date])
                dates_inds_list.append(idx_date)

            if neu_vs_date_mat is None:
                continue
            else:
                neu_vs_date_mat = np.array(neu_vs_date_mat)
            dates_corr_mat = np.corrcoef(neu_vs_date_mat)  # Pearson correlation coefficient
            plt.figure(figsize=(10, 10))
            ax = sns.heatmap(dates_corr_mat, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
            plt.title(f'Correlation Matrix across dates for flavor {flavor_key}')
            relevent_combined_labels = [combined_labels[i] for i in dates_inds_list]
            ax.set_xticklabels(relevent_combined_labels, rotation=0)
            ax.set_yticklabels(relevent_combined_labels, rotation=0)
            plt.savefig(os.path.join(animal_directory, f'Corr_mat_vs_date_for_{flavor_key}.png'))
            # spio.savemat(os.path.join(animal_directory, f'Corr_mat_vs_date_for_{flavor_key}.mat'),
            #              {'correlation_matrix': dates_corr_mat,
            #               'combined_labels': relevent_combined_labels})

            # Calculate the correlation matrix between each of the experiment part
            main_parts = ['first', 'ongoing_batch', 'ongoing_random']
            #main_parts = ['first', 'ongoing_batch', 'ongoing_random']
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
            plt.savefig(os.path.join(animal_directory, f'Parts_correlation_{flavor_key}.png'))
            # spio.savemat(os.path.join(animal_directory, f'Parts_correlation_{flavor_key}.mat'),
            #              {'correlation_matrix_parts': corr_mat_parts_1_flav,
            #               'main_parts': main_parts})

            corr_mats_parts_per_flavor[idx_flavor] = corr_mat_parts_1_flav

        return corr_mats_parts_per_flavor, flavors_corr_mat_list_parts

    def corr_parts_flavors_context_all_animals(self):

        main_parts = ['first', 'ongoing_batch', 'ongoing_random']
        corr_mats_parts_per_flavor_all_animals = [[] for _ in range(len(self.flavors2num))]
        corr_mats_flavors_per_parts_all_animals = [[] for _ in range(len(main_parts))]
        for animal_num in self.animals:
            animal = animal_num[:-2]
            corr_mats_parts_per_flavor, flavors_corr_mat_list_parts = self.__correlation_mat_flavors_context_1_animal(animal)
            for i in range(len(self.flavors2num)):
                    corr_mats_parts_per_flavor_all_animals[i].append(corr_mats_parts_per_flavor[i])
            for i in range(len(main_parts)):
                for j in range(len(flavors_corr_mat_list_parts[i])):
                    corr_mats_flavors_per_parts_all_animals[i].append(flavors_corr_mat_list_parts[i][j])

        flavors2num_tmp = {'g': 1, 's': 2, 'q': 3}
        for idx_flavor, flavor in enumerate(flavors2num_tmp):
            corr_parts_matrices = corr_mats_parts_per_flavor_all_animals[idx_flavor]
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
            plt.title(f'Correlation Matrix across experiment parts -  All animals - flavor {flavor}')
            ax.set_xticklabels(main_parts, rotation=0)
            ax.set_yticklabels(main_parts, rotation=0)
            plt.savefig(os.path.join(self.res_main_directory, f'Corr_parts_all_animals_flavor_{flavor}.png'))

        fig, axes = plt.subplots(nrows=1, ncols=len(main_parts), figsize=(20, 7))
        for idx, (title, mat_list) in enumerate(zip(self.main_parts, corr_mats_flavors_per_parts_all_animals)):
            if title == 'first' or title == 'ongoing_random':
                flavors2num_tmp = {'g': 1, 's': 2, 'q': 3}
            elif title == 'ongoing_batch':
                flavors2num_tmp = {'g': 1, 's': 2, 'q': 3, 'f': 4}
            mean_conf, err_conf = combine_list_of_conf_mat(mat_list, flavors2num_tmp)
            cax = axes[idx].matshow(mean_conf, cmap='coolwarm')
            axes[idx].set_title(title)

            # Set axis labels
            labels = list(flavors2num_tmp.keys())
            axes[idx].set_xticks(np.arange(len(labels)))  # Set the positions for x ticks
            axes[idx].set_yticks(np.arange(len(labels)))
            axes[idx].set_xlabel('Predicted labels')
            axes[idx].set_ylabel('True labels')
            axes[idx].set_xticklabels(labels)
            axes[idx].set_yticklabels(labels)

            # Annotate each cell with the mean and std values
            for (i, j), val in np.ndenumerate(mean_conf):
                axes[idx].text(j, i, f'{mean_conf[i, j]:.2f}\n±{err_conf[i, j]:.2f}', ha='center', va='center',
                               color='black')

        plt.tight_layout()
        fig.colorbar(cax, ax=axes.ravel().tolist(), orientation='vertical')
        title = f"Correlation matrix between flavors per parts"
        fig.suptitle(title, fontsize=16)
        plt.savefig(os.path.join(self.res_main_directory, title + ".png"))



    def analysis_per_flavor_per_parts_vs_time_windows(self, analysis, num_flavors_relevant=None):  # flavors context, success classification
        # analysis options: 'open_gates', 'acc_per_flavors'
        self.curr_method = 'percent_open_gates_per_flavor_per_parts_vs_time_windows_' + analysis
        if not self.context == 'flavors' or not self.classification_type == 'success':
            raise ValueError("Not suitable context for this method")
        self.run_type = "flavors_context"

        time_windows = ["_-3_-1", "_2_4", "_6_8"]
        main_parts = ['first', 'ongoing_batch', 'ongoing_random']
        fig, axes = plt.subplots(len(time_windows), len(main_parts), figsize=(25, 10))
        if analysis == 'open_gates':
            fig.suptitle(f'Open gate percentage for Each Flavor, for different time windows and experiment part - {str(num_flavors_relevant)} flavors', fontsize=16)
        elif analysis == 'acc_per_flavors':
            fig.suptitle(f'Accuracies for Each Flavor, for different time windows and experiment part - {str(num_flavors_relevant)} flavors', fontsize=16)


        for row_idx, times in enumerate(time_windows):
            self.run_type = "flavors_context" + times + "_lambda_options" #todo

            parts_sublists = self.__load_results(include_chance_level=False, num_flavors_relevant=num_flavors_relevant)
            percent_open_gate_per_flavors_per_parts = []

            for i, data_list in enumerate(parts_sublists):

                # Initialize a dictionary to hold sums and counts for each flavor
                flavor_data = defaultdict(lambda: {'sum': 0, 'count': 0})
                i=0
                # Iterate through each tuple
                for open_gate, flavors in data_list:
                    # Iterate through each flavor and its corresponding number
                    for flavor, number in zip(flavors, open_gate):
                        # if len(flavors)!= 3:
                        #     continue
                        # Update the cumulative sum and count for each flavor
                        flavor_data[flavor]['sum'] += number
                        flavor_data[flavor]['count'] += 1
                    i+=1

                # Calculate the mean for each flavor
                flavors_means = {flavor: data['sum'] / data['count'] for flavor, data in flavor_data.items()}
                percent_open_gate_per_flavors_per_parts.append(flavors_means)

            # Now plot for each part in the specific row for the current time window
            for col_idx, (flavor_means, part) in enumerate(
                    zip(percent_open_gate_per_flavors_per_parts, self.main_parts)):
                # Accessing the correct subplot for current time window and part
                ax = axes[row_idx, col_idx] if len(time_windows) > 1 else axes[
                    col_idx]  # Adjust based on whether you have one or multiple time windows
                flavors = list(flavor_means.keys())
                means = [flavor_means[flavor] for flavor in flavors]

                if analysis == 'open_gates':
                    color = 'skyblue'
                elif analysis == 'acc_per_flavors':
                    color = 'tomato'
                ax.bar(flavors, means, color=color)
                ax.set_xlabel('Flavors')
                if analysis == 'open_gates':
                    ax.set_ylabel('Open gates %')
                    ax.set_ylim(0.3, 1)
                elif analysis == 'acc_per_flavors':
                    ax.set_ylabel('Accuracy %')
                    ax.set_ylim(0, 1)
                start_time = times.split('_')[1]; end_time = times.split('_')[2]
                ax.set_title(f'{start_time} to {end_time} sec, {part}')
                ax.tick_params(axis='x', rotation=0)  # Adjust rotation if necessary



            # Adjust layout for a better fit
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        if analysis == 'open_gates':
            plt.savefig(os.path.join(self.res_main_directory,
                                     f'Open gate percentage for Each Flavor parts and time windows - {str(num_flavors_relevant)} flavors.png'))
        elif analysis == 'acc_per_flavors':
            plt.savefig(os.path.join(self.res_main_directory,
                                     f'Accuracies for Each Flavor parts and time windows - {str(num_flavors_relevant)} flavors.png'))































