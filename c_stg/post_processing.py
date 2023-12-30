# Imports
from flavors.data_params import data_origanization_params
from flavors.utils import hyperparameters_chosen_extraction, init_optimizer, init_criterion
from flavors.data_processing import DataProcessor, DataContainer
from visual import visual_results
from params import Params
import torch
import models
from training import train, test_process, get_prob_alpha
import numpy as np
import torch.utils.data as data_utils
import scipy.io as spio
import random


def post_process_flow(directory_name):

    # Parameters
    params = Params()
    params = data_origanization_params(params)
    params.post_process_mode = True  # flag for post-processing
    params.infer_directory = params.result_directory + directory_name

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
    # input_dim = no. of explanatory features
    params.input_dim = data.explan_feat.shape[-1]
    # input_dim = no. of contextual features
    params.param_dim = 1
    #
    params.output_dim = 1
    Container = DataContainer(params, data, fold=0)  # only one fold option
    train_Dataloader, test_Dataloader = Container.get_Dataloaders(params)

    # Training
    # Load model architecture
    model = models.__dict__[params.ML_model_name]\
        (params.input_dim, hidden_dim, params.output_dim, params.param_dim, hyper_hidden_dim,
         params.dropout, sigma=params.sigma, include_B_in_input=params.include_B_in_input,
         non_param_stg=params.non_param_stg, train_sigma=params.train_sigma)

    model = model.to(params.device).float()
    criterion = init_criterion()
    optimizer = init_optimizer(model, learning_rate)

    acc_train_array, loss_train_array, acc_test_array, loss_test_array = \
        train(params, model, train_Dataloader, test_Dataloader, criterion, optimizer,
              stg_regularizer, final_test=params.post_process_mode)

    model.eval()
    unique_r = np.unique(Container.rte)
    alpha_vals = np.zeros((Container.xtr.shape[1], len(unique_r)))
    acc_vals_per_r = np.zeros(len(unique_r))
    ri = 0
    for rval in np.unique(Container.rte):
        alpha_vals[:, ri] = get_prob_alpha(params, model, np.array(rval).reshape(-1, 1))
        inds = [i for i, x in enumerate(Container.rte == rval) if x]
        x_test_tmp = Container.xte[inds, :]
        r_test_tmp = Container.rte[inds].reshape(-1, 1)
        y_test_tmp = Container.yte[inds].reshape(-1, 1)
        test_set_tmp = data_utils.TensorDataset(torch.tensor(x_test_tmp), torch.tensor(y_test_tmp), torch.tensor(r_test_tmp))
        test_dataloader_tmp = torch.utils.data.DataLoader(test_set_tmp, batch_size=params.batch_size, shuffle=False)
        acc_dev, _ = test_process(params, model, test_dataloader_tmp, criterion, stg_regularizer)
        acc_vals_per_r[ri] = acc_dev
        ri += 1

    spio.savemat(filename,
                 {'acc_train_array': acc_train_array, 'loss_train_array': loss_train_array,
                  'acc_test_array': acc_test_array, 'loss_test_array':loss_test_array,
                  'unique_r': unique_r, 'alpha_vals': alpha_vals, 'acc_vals_per_r': acc_vals_per_r})

    mat_name = hyperparameter_combination + "_Final_check" + ".mat"
    # chance level is calculated in DataProcessor
    visual_results(params.infer_directory, mat_name, params.chance_level*100)

