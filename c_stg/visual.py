import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio


# Visualization of the result after post-processing
def visual_results(path, mat_name, chance_level):
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
    chance_level = np.ones(acc_test_array[0].shape)*chance_level
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
    alpha_vals = mat_data.get('alpha_vals', None)
    time_valus = np.linspace(-4, 16, alpha_vals.shape[1])
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.05], height_ratios=[2, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    cax1 = ax1.imshow(alpha_vals, aspect='auto', extent=[time_valus[0], time_valus[-1], 0, alpha_vals.shape[0]])
    ax1.set_title('Alpha Values')
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
    plt.show()
    plt.savefig(os.path.join(path, 'Accuracy_per_r_&_Alphas_values.png'))

    # Accuracy values per r + Alphas values organized
    ic = np.argsort(alpha_vals[:, -1])
    sorted_alpha_vals = alpha_vals[ic, :]
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.05], height_ratios=[2, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    cax1 = ax1.imshow(sorted_alpha_vals, aspect='auto', extent=[time_valus[0], time_valus[-1], 0, alpha_vals.shape[0]])
    ax1.set_title('Alpha Values')
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
    plt.show()
    plt.savefig(os.path.join(path, 'Accuracy_per_r_&_Alphas_values_ORGANIZED.png'))

