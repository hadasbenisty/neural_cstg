# Imports
from c_stg.data_processing import DataContainer
from c_stg.params import Params_config
import torch
import c_stg.models
from c_stg.training import train, test_process, get_prob_alpha
import numpy as np
import torch.utils.data as data_utils
import scipy.io as spio
import random
from c_stg.utils import import_per_data_type
from sklearn.metrics import confusion_matrix


def post_process_flow(data_type, directory_name, cstg_args={}, data_args={}):
    # Specific imports
    (acc_score, set_seed, init_criterion, init_optimizer, DataProcessor, data_origanization_params, Data_params,
     hyperparameters_chosen_extraction, visual_results) = \
        (import_per_data_type(data_type))

    # Parameters
    params = Params_config(data_type, cstg_kwargs=cstg_args, data_kwargs=data_args)
    params.post_process_mode = True  # flag for post-processing
    params.infer_directory = params.result_directory + directory_name
    params = data_origanization_params(params)

    if params.manual_random_seed != -1:  # -1 for no setting
        random.seed(params.manual_random_seed)
        torch.manual_seed(params.manual_random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    filename, hidden_dim, hyper_hidden_dim, learning_rate, stg_regularizer, hyperparameter_combination\
        = hyperparameters_chosen_extraction(params)
    print(filename)

    # Data
    data = DataProcessor(params)
    params = data.params
    # input_dim = no. of explanatory features
    params.input_dim = data.explan_feat.shape[-1]
    # input_dim = no. of contextual features
    params.param_dim = 1
    #
    num_labels = len(np.unique(np.array(data.output_label)))
    if num_labels == 2:
        params.output_dim = 1
    else:
        params.output_dim = num_labels

    # uneffective_flag = True
    # while uneffective_flag:
        # set_seed(int(time.time()))
    Container = DataContainer(params, data, fold=0)  # only one fold option
    train_Dataloader, test_Dataloader = Container.get_Dataloaders(params)

    # Training
    # Load model architecture
    model = c_stg.models.__dict__[params.ML_model_name]\
        (params.input_dim, hidden_dim, params.output_dim, params.param_dim, hyper_hidden_dim,
         params.dropout, sigma=params.sigma, include_B_in_input=params.include_B_in_input,
         non_param_stg=params.non_param_stg, train_sigma=params.train_sigma, classification=params.classification_flag)

    model = model.to(params.device).float()
    criterion = init_criterion(params)
    optimizer = init_optimizer(model, learning_rate)


    acc_train_array, loss_train_array, acc_test_array, loss_test_array, uneffective_flag = \
        train(params, model, train_Dataloader, test_Dataloader, criterion, optimizer,
              stg_regularizer, acc_score)

    model.eval()
    unique_r = np.unique(Container.rte)
    # alpha_vals = np.zeros((Container.xtr.shape[1], len(unique_r)))
    mu_vals = np.zeros((Container.xtr.shape[1], len(unique_r)))
    if params.ML_model_name == "fc_stg_layered_param_modular_model_sigmoid_extension":
        w_vals = np.zeros((Container.xtr.shape[1], len(unique_r)))
    else:
        w_vals = []
    acc_vals_per_r = np.zeros(len(unique_r))
    #conf_mat_per_r = []
    ri = 0
    for rval in np.unique(Container.rte):
        # alpha_vals[:, ri] = get_prob_alpha(params, model, np.array(rval).reshape(-1, 1))
        if params.ML_model_name == "fc_stg_layered_param_modular_model_sigmoid_extension":
            mu_vals[:, ri], w_vals[:, ri] = \
                get_prob_alpha(params, model, np.array(rval).reshape(-1, 1))
        elif params.ML_model_name == "fc_stg_layered_param_modular_model_sigmoid":
            mu_vals[:, ri] = get_prob_alpha(params, model, np.array(rval).reshape(-1, 1))
        inds = [i for i, x in enumerate(Container.rte == rval) if x]
        x_test_tmp = Container.xte[inds, :]
        r_test_tmp = Container.rte[inds].reshape(-1, 1)
        y_test_tmp = Container.yte[inds].reshape(-1, 1)
        test_set_tmp = data_utils.TensorDataset(torch.tensor(x_test_tmp),
                                                torch.tensor(y_test_tmp), torch.tensor(r_test_tmp))
        test_dataloader_tmp = torch.utils.data.DataLoader(test_set_tmp, batch_size=params.batch_size, shuffle=False)
        acc_dev, _, true_labels, predicted_labels =\
            test_process(params, model, test_dataloader_tmp, criterion, stg_regularizer)
        acc_vals_per_r[ri] = acc_dev
        #conf_mat_per_r.append(confusion_matrix(true_labels.flatten().cpu(), predicted_labels.cpu()))
        ri += 1

    spio.savemat(filename,
                 {'acc_train_array': acc_train_array, 'loss_train_array': loss_train_array,
                  'acc_test_array': acc_test_array, 'loss_test_array':loss_test_array,
                  'unique_r': unique_r, 'mu_vals': mu_vals,
                  'w_vals': w_vals, 'acc_vals_per_r': acc_vals_per_r})

    mat_name = hyperparameter_combination + "_Final_check" + ".mat"
    # chance level is calculated in DataProcessor
    visual_results(params.infer_directory, mat_name, params)


# if __name__ == '__main__':
#     # name = ['2024_01_06_22_05_35_animal_4575_date_03_14_19_flavors',
#     #         '2024_01_06_23_15_21_animal_4575_date_03_19_19_flavors',
#     #         '2024_01_07_00_43_30_animal_4575_date_03_31_19_flavors',
#     #         '2024_01_07_01_43_37_animal_4575_date_04_03_19_flavors',
#     #         '2024_01_07_03_12_30_animal_4575_date_04_07_19_flavors',
#     #         '2024_01_07_04_40_12_animal_4575_date_04_11_19_flavors',
#     #         '2024_01_07_05_55_30_animal_4575_date_04_15_19_flavors']
#     # date = ['03_14_19', '03_19_19', '03_31_19', '04_03_19', '04_07_19', '04_11_19', '04_15_19']
#     name = '2024_01_20_23_54_03_animal_4458_date_01_22_19_success'
#     arguments_dict = {'animal': '4458', 'date': '01_22_19'}
#     post_process_flow(name, **arguments_dict)