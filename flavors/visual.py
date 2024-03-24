import os
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import pandas as pd
import seaborn as sns



# Visualization of the result after post-processing
def visual_results(path, mat_name, params):
    mat_path = os.path.join(path, mat_name)
    mat_data = spio.loadmat(mat_path)

    # Accuracies
    acc_test_array = mat_data.get('acc_test_array', None)
    acc_train_array = mat_data.get('acc_train_array', None)
    plt.figure()
    plt.plot(acc_test_array[0], label='Test accuracy', color='blue')
    plt.plot(acc_train_array[0], label='Train accuracy', color='orange')
    plt.legend()
    plt.title('Accuracy of test and train datasets VS epochs')
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(path, 'Accuracies.png'))

    # Relative accuracies: accuracy - chance_level
    plt.figure()
    chance_level = np.ones(acc_test_array[0].shape)*params.chance_level*100
    chance_level[0] = 0
    plt.plot(acc_test_array[0]-chance_level, label='Test accuracy', color='blue')
    plt.plot(acc_train_array[0]-chance_level, label='Train accuracy', color='orange')
    plt.legend()
    plt.title('Relative - Accuracy of test and train datasets VS epochs')
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy - chance_level')
    plt.savefig(os.path.join(path, 'Relative_accuracies.png'))

    # Losses
    loss_test_array = mat_data.get('loss_test_array', None)
    loss_train_array = mat_data.get('loss_train_array', None)
    plt.figure()
    plt.plot(loss_test_array[0], label='Test loss', color='blue')
    plt.plot(loss_train_array[0], label='Train loss', color='orange')
    plt.legend()
    plt.title('Loss value of test and train datasets VS epochs')
    plt.xlabel('Epoch number')
    plt.ylabel('Loss value')
    plt.savefig(os.path.join(path, 'Losses.png'))

    # Accuracy values per r + Alphas values
    acc_vals_per_r = mat_data.get('acc_vals_per_r', None)
    mu_vals = mat_data.get('mu_vals', None)
    fig_names_mus = ['Accuracy_per_r_&_MUs_values.png', 'Accuracy_per_r_&_MUs_values_ORGANIZED.png']
    ic_mus = np.argsort(mu_vals[:, 0])
    sorted_mu_vals = mu_vals[ic_mus, :]
    mus = [mu_vals, sorted_mu_vals]

    if params.ML_model_name == "fc_stg_layered_param_modular_model_sigmoid_extension":
        w_vals = mat_data.get('w_vals', None)
        fig_names_gates = ['Accuracy_per_r_&_Weights_values.png', 'Accuracy_per_r_&_Weights_values_ORGANIZED.png']
        ic_gates = np.argsort(w_vals[:, 0])
        sorted_w_vals = w_vals[ic_gates, :]
        weights = [w_vals, sorted_w_vals]

        alphas = [mus, weights]
        fig_names = [fig_names_mus, fig_names_gates]
        titles = ["Mu", "Weights"]
    else:
        alphas = [mus]
        fig_names = [fig_names_mus]
        titles = ["Mu"]

    if params.context_key == 'flavors':
        for fig_name, alpha, title in zip(fig_names, alphas, titles):
            if params.ML_model_name != "fc_stg_layered_param_modular_model_sigmoid_extension" and title == "Weights":
                continue
            for sub_fig_name, sub_alpha in zip(fig_name, alpha):
                info_df = pd.read_excel(params.info_excel_path, sheet_name=params.animal2sheet_num[params.animal])
                flavors_list = info_df[info_df['folder'] == params.date]['flavors'].iloc[0].split('_')

                fig = plt.figure(figsize=(12, 8))
                gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.05], height_ratios=[2, 1])

                ax1 = fig.add_subplot(gs[0, 0])
                sns.heatmap(sub_alpha, ax=ax1, cbar_ax=fig.add_subplot(gs[0, 1]))
                ax1.set_title(title)
                ax1.set_xticks(np.arange(len(flavors_list)) + 0.5)
                ax1.set_xticklabels(flavors_list)

                # Creating a bar graph
                ax2 = fig.add_subplot(gs[1, 0])
                ax2.bar(flavors_list, acc_vals_per_r.flatten(), color='tomato')  # You can choose the color
                ax2.set_title('Accuracy Values Per R')
                ax2.grid(True)

                plt.tight_layout()
                plt.savefig(os.path.join(path, sub_fig_name))

    elif params.context_key == 'time':

        for fig_name, alpha, title in zip(fig_names, alphas, titles):
            if params.ML_model_name != "fc_stg_layered_param_modular_model_sigmoid_extension" and title == "Weights":
                continue
            for sub_fig_name, sub_alpha in zip(fig_name, alpha):
                time_valus = np.linspace(params.start_time, params.end_time, sub_alpha.shape[1])
                fig = plt.figure(figsize=(12, 8))
                gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.05], height_ratios=[2, 1])

                ax1 = fig.add_subplot(gs[0, 0])
                cax1 = ax1.imshow(sub_alpha, aspect='auto', extent=[time_valus[0], time_valus[-1], 0, sub_alpha.shape[0]])
                ax1.set_title(title)
                ax1.axvline(x=0, color='red', linestyle='--')  # Adding vertical line at 0 for Alpha Vals
                ax_colorbar = fig.add_subplot(gs[0, 1])
                plt.colorbar(cax1, cax=ax_colorbar)

                ax2 = fig.add_subplot(gs[1, 0])
                ax2.plot(time_valus.flatten(), acc_vals_per_r.flatten())
                ax2.set_title('Accuracy Values Per R')
                ax2.set_xlim(time_valus[0], time_valus[-1])
                ax2.axvline(x=0, color='red', linestyle='--')
                ax2.grid(True)

                plt.tight_layout()
                plt.savefig(os.path.join(path, sub_fig_name))

        # # confusion matrix
        # for time_point in conf_mat_times:
        #     diff = np.abs(time_valus - time_point)
        #     closest_idx = np.argmin(diff)
        #     conf_mat_per_r = mat_data.get('conf_mat_per_r', None)
        #     current_conf_mat = conf_mat_per_r[closest_idx]
        #
        #     # Visualize the confusion matrix
        #     plt.figure(figsize=(8, 6))
        #     ax = sns.heatmap(current_conf_mat, annot=True, fmt='g', cmap='Blues')
        #     plt.xlabel('Predicted Labels')
        #     plt.ylabel('True Labels')
        #     if current_conf_mat.shape[0] == 4:
        #         ax.set_xticklabels(['fail', 's', 'q', 'g'], rotation=0)
        #         ax.set_yticklabels(['fail', 's', 'q', 'g'], rotation=0)
        #     if current_conf_mat.shape[0] == 5:
        #         ax.set_xticklabels(['fail', 's', 'q', 'g', 'f'], rotation=0)
        #         ax.set_yticklabels(['fail', 's', 'q', 'g', 'f'], rotation=0)
        #
        #     plt.title(f'Confusion Matrix for time {time_point}')
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(path, f'Confusion Matrix for time {time_point}'))



# path = '../results/2024_01_01_11_17_32_animal_4575_date_03_05_19_success'
# mat_name = 'c-stg_hidden[1000]_lr0.001_lam0.05_Final_check.mat'
# from params import Params
# params = Params()
# params.end_time = 16
# params.start_time = -4
# params.chance_level = 0.65
# params.date = '03_19_19'
# visual_results(path, mat_name, params)