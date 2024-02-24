import os
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio


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

    mus_vec_r = []
    for i in np.linspace(0, len(mu_vals)-1, len(mu_vals)).astype(int):
        mus_vec_r.append(mu_vals[i])

    fig, axs = plt.subplots(mu_vals.shape[1])
    for ii in np.linspace(0, mu_vals.shape[1] - 1, mu_vals.shape[1]).astype(int):
        axs[ii].bar(np.linspace(1, mu_vals[:, 0].size, mu_vals[:, 0].size).astype(int), mu_vals[:, ii], label='Features Weights')
        axs[ii].set_xlabel('Feature Number #')
        axs[ii].set_ylabel('Weight')
        axs[ii].set_title(['Weights as a function of numbers for context no.', str(ii+1)])
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'feature_weights.png'))
