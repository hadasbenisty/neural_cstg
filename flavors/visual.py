import os
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import pandas as pd
import seaborn as sns
from flavors.utils import find_best_hyper_comb
from flavors.analysis_results.utils import extract_value


# Visualization of the result after post-processing
def post_process_visualization(infer_directory):
    log_file = os.path.join(infer_directory, "log.txt")
    ML_model_name = extract_value(log_file, "ML_model_name", occurrence=1)
    context_key = extract_value(log_file, "context_key", occurrence=1)
    best_comb = find_best_hyper_comb(infer_directory, key="nn_acc_dev")

    acc_dev_list = []
    acc_train_list = []
    loss_dev_list = []
    loss_train_list = []
    acc_vals_per_r_list = []
    mu_vals_list = []
    w_vals_list = []

    best_comb_directory = os.path.join(infer_directory, best_comb)
    folds_num = int(extract_value(log_file, "folds_num", occurrence=1))
    for foldNum in range(folds_num):
        mat_path = os.path.join(best_comb_directory, f"selfold{foldNum}.mat")
        mat_data = spio.loadmat(mat_path)
        acc_dev_array = mat_data.get('dev_acc_array', None)
        acc_train_array = mat_data.get('train_acc_array', None)
        loss_dev_array = mat_data.get('dev_loss_array', None)
        loss_train_array = mat_data.get('train_loss_array', None)
        acc_vals_per_r = mat_data.get('acc_vals_per_r', None)
        mu_vals = mat_data.get('mu_vals', None)
        if ML_model_name == "fc_stg_layered_param_modular_model_sigmoid_extension":
            w_vals = mat_data.get('w_vals', None)

        acc_dev_list.append(acc_dev_array[0])
        acc_train_list.append(acc_train_array[0])
        loss_dev_list.append(loss_dev_array[0])
        loss_train_list.append(loss_train_array[0])
        acc_vals_per_r_list.append(acc_vals_per_r[0])
        mu_vals_list.append(mu_vals)
        if ML_model_name == "fc_stg_layered_param_modular_model_sigmoid_extension":
            w_vals_list.append(w_vals)

    acc_dev_stack = np.stack(acc_dev_list)
    acc_train_stack = np.stack(acc_train_list)
    loss_dev_stack = np.stack(loss_dev_list)
    loss_train_stack = np.stack(loss_train_list)
    acc_vals_per_r_stack = np.stack(acc_vals_per_r_list)
    mu_vals_stack = np.stack(mu_vals_list)
    if ML_model_name == "fc_stg_layered_param_modular_model_sigmoid_extension":
        w_vals_stack = np.stack(w_vals_list)

    # Accuracies
    plot_2_graphs(acc_dev_stack, acc_train_stack, 'Accuracy of Develop and Train Datasets vs Epochs',
                  'Epoch Number', 'Accuracy', 'Dev accuracy', 'Train accuracy', infer_directory,
                  'Accuracies.png')

    # Relative accuracies: accuracy - chance_level
    chance_level = float(extract_value(log_file, "chance_level", occurrence=1))
    chance_level = np.ones(acc_dev_stack.shape[1]) * chance_level * acc_dev_stack.shape[1]
    plot_2_graphs(acc_dev_stack, acc_train_stack, 'Relative - Accuracy of test and train datasets VS epochs',
                  'Epoch Number', 'Accuracy - chance_level', 'Test accuracy', 'Train accuracy',
                  infer_directory,'Relative_accuracies.png', chance_level=chance_level)

    # Losses
    plot_2_graphs(loss_dev_stack, loss_train_stack, 'Loss value of test and train datasets VS epochs',
                  'Epoch Number', 'Loss value', 'Test loss', 'Train loss',
                  infer_directory,'Losses.png')


    # Accuracy values per r + Alphas values
    acc_vals_per_r = np.mean(acc_vals_per_r_stack, axis=0)
    acc_vals_per_r_error = np.std(acc_vals_per_r_stack, axis=0) / np.sqrt(acc_vals_per_r_stack.shape[0])
    # Take randomly the first from the folds
    mu_vals = mu_vals_stack[0]
    if ML_model_name == "fc_stg_layered_param_modular_model_sigmoid_extension":
        w_vals = w_vals_stack[0]

    fig_names_mus = ['Accuracy_per_r_&_MUs_values.png', 'Accuracy_per_r_&_MUs_values_ORGANIZED.png']
    ic_mus = np.argsort(mu_vals[:, 0])
    sorted_mu_vals = mu_vals[ic_mus, :]
    mus = [mu_vals, sorted_mu_vals]

    if ML_model_name == "fc_stg_layered_param_modular_model_sigmoid_extension":
        #w_vals = mat_data.get('w_vals', None)
        fig_names_gates = ['Accuracy_per_r_&_Weights_values.png', 'Accuracy_per_r_&_Weights_values_ORGANIZED.png']
        ic_gates = np.argsort(w_vals[:, 0])
        sorted_w_vals = w_vals[ic_gates, :]
        weights = [w_vals, sorted_w_vals]

        alphas = [mus, weights]
        fig_names = [fig_names_mus, fig_names_gates]
        titles = ["Mu (for 1 fold)", "Weights"]
    else:
        alphas = [mus]
        fig_names = [fig_names_mus]
        titles = ["Mu"]


    if context_key == 'flavors':
        import ast
        animal = extract_value(log_file, "animal", occurrence=1)
        date = extract_value(log_file, "date", occurrence=1)
        info_excel_path = '../data/animals_db_selected.xlsx'
        flavors2num = ast.literal_eval(extract_value(log_file, "flavors2num", occurrence=1))
        animal2sheet_num = ast.literal_eval(extract_value(log_file, "animal2sheet_num", occurrence=1))
        info_df = pd.read_excel(info_excel_path, sheet_name=animal2sheet_num[animal])
        flavors_list = info_df[info_df['folder'] == date]['flavors'].iloc[0].split('_')
        # Sort flavors based on the values in the flavors2num dictionary
        flavors = sorted(flavors_list, key=lambda flavor: flavors2num[flavor])

        for fig_name, alpha, title in zip(fig_names, alphas, titles):
            if ML_model_name != "fc_stg_layered_param_modular_model_sigmoid_extension" and title == "Weights":
                continue
            for sub_fig_name, sub_alpha in zip(fig_name, alpha):

                fig = plt.figure(figsize=(12, 8))
                gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.05], height_ratios=[2, 1])

                ax1 = fig.add_subplot(gs[0, 0])
                sns.heatmap(sub_alpha, ax=ax1, cbar_ax=fig.add_subplot(gs[0, 1]))
                ax1.set_title(title)
                ax1.set_xticks(np.arange(len(flavors_list)) + 0.5)
                ax1.set_xticklabels(flavors_list)

                # Creating a bar graph
                ax2 = fig.add_subplot(gs[1, 0])
                ax2.bar(flavors_list, acc_vals_per_r.flatten(), color='tomato', yerr=acc_vals_per_r_error, capsize=5)  # You can choose the color
                ax2.set_title('Accuracy Values Per R')
                ax2.grid(True)

                plt.tight_layout()
                plt.savefig(os.path.join(infer_directory, sub_fig_name))

    if context_key == 'time':
        start_time = float(extract_value(log_file, "start_time", occurrence=1))
        drop_time = float(extract_value(log_file, "drop_time", occurrence=1))
        start_time = start_time + drop_time
        end_time = float(extract_value(log_file, "end_time", occurrence=1))

        for fig_name, alpha, title in zip(fig_names, alphas, titles):
            if ML_model_name != "fc_stg_layered_param_modular_model_sigmoid_extension" and title == "Weights":
                continue
            for sub_fig_name, sub_alpha in zip(fig_name, alpha):

                time_valus = np.linspace(start_time, end_time, sub_alpha.shape[1])
                fig = plt.figure(figsize=(12, 8))
                gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.05], height_ratios=[2, 1])

                ax1 = fig.add_subplot(gs[0, 0])
                cax1 = ax1.imshow(sub_alpha, aspect='auto', extent=[time_valus[0], time_valus[-1], 0, sub_alpha.shape[0]])
                ax1.set_title(title)
                ax1.axvline(x=0, color='red', linestyle='--')  # Adding vertical line at 0 for Alpha Vals
                ax_colorbar = fig.add_subplot(gs[0, 1])
                plt.colorbar(cax1, cax=ax_colorbar)

                ax2 = fig.add_subplot(gs[1, 0])
                ax2.fill_between(time_valus.flatten(), (acc_vals_per_r - acc_vals_per_r_error).flatten(),
                                 (acc_vals_per_r + acc_vals_per_r_error).flatten(), alpha=0.5, color='blue')
                ax2.plot(time_valus.flatten(), acc_vals_per_r.flatten(), color='darkblue')
                ax2.set_title('Accuracy Values Per R - Test dataset')
                ax2.set_xlim(time_valus[0], time_valus[-1])
                ax2.axvline(x=0, color='red', linestyle='--')
                ax2.grid(True)

                plt.tight_layout()
                plt.savefig(os.path.join(infer_directory, sub_fig_name))

    mat_path = os.path.join(infer_directory, best_comb + "_processed.mat")
    spio.savemat(mat_path,
                 {'acc_vals_per_r_stacked': acc_vals_per_r_stack})

def plot_2_graphs(acc_stack1, acc_stack2, title, xlabel, ylabel, legend1, legend2, path, filename, chance_level=0):
    """
    Plots two graphs for accuracy stacks with error margins.

    Parameters:
    - acc_stack1, acc_stack2: Numpy arrays for the first and second stacked data.
    - title: Title for the graphs.
    - xlabel, ylabel: Labels for the x-axis and y-axis.
    - legend1, legend2: Legend labels for the first and second graph.
    - path: Directory path to save the figure.
    - filename: Name of the file to save the figure.
    - chance_level: make the graphs relative to some baseline
    """

    # Calculate means and standard errors for both stacks
    mean1 = np.mean(acc_stack1, axis=0)
    err1 = np.std(acc_stack1, axis=0) / np.sqrt(acc_stack1.shape[0])
    mean2 = np.mean(acc_stack2, axis=0)
    err2 = np.std(acc_stack2, axis=0) / np.sqrt(acc_stack2.shape[0])

    # Relative
    mean1 = mean1 - chance_level
    mean2 = mean2 - chance_level

    # Create figure and axis
    fig, ax = plt.subplots()
    ax.fill_between(range(len(mean1)), mean1 - err1 / 2, mean1 + err1 / 2, alpha=0.5, color='blue')
    ax.fill_between(range(len(mean2)), mean2 - err2 / 2, mean2 + err2 / 2, alpha=0.5, color='orange')
    ax.plot(mean1, label=legend1, color='darkblue')
    ax.plot(mean2, label=legend2, color='darkorange')

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(os.path.join(path, filename))

# animals = ['4458_0', '4575_1', '4754_2', '4756_3', '4880_4', '4882_5']
# for animal in animals:
#     directory = '../results/'+animal+'/flavors_classification'
#     print(animal)
#     for subdir in os.listdir(directory):
#         if not subdir.endswith('png'):
#             comb = os.path.join(directory,subdir)
#             post_process_visualization(comb)