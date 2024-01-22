# Imports
from c_stg.params import Params_config
import os
import scipy.io as spio
import torch.utils.data as data_utils
from c_stg.training import *
import c_stg.models
import time
from sklearn.linear_model import LogisticRegression
from c_stg.post_processing import post_process_flow
from c_stg.utils import import_per_data_type
from c_stg.data_processing import DataContainer


#def main_workflow(data_type='flavors', **kwargs):
def main_workflow(data_type='flavors', cstg_args={}, data_args={}):
    # Specific imports
    (acc_score, set_seed, init_criterion, init_optimizer, DataProcessor, data_origanization_params, Data_params,
     hyperparameters_chosen_extraction, visual_results) = \
        (import_per_data_type(data_type))

    # Parameters
    #params = Params_config(data_type, **kwargs)  # kwargs are arguments for Data_params
    params = Params_config(data_type, cstg_kwargs=cstg_args, data_kwargs=data_args)
    params = data_origanization_params(params)  # add res_directory property to params

    # Write running parameters to a text file
    os.makedirs(params.res_directory, exist_ok=True)
    print('Writing results to %s\n' % params.res_directory)
    with open(os.path.join(params.res_directory, 'log.txt'), 'w') as f:
        f.write(''.join(["%s = %s\n" % (k, v) for k, v in params.__dict__.items()]))

    # params.chance_level and data.use_flag are calculated in DataProcessor
    # cross validation k fold split is done in DataProcessor
    data = DataProcessor(params)
    params = data.params
    if not data.use_flag:  # the data not suitable
        return

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

    if params.include_linear_model:
        for c_value in params.inverse_regularization:
            train_acc = []
            dev_acc = []
            print(f"c_value is {str(c_value)}")
            for fold in range(params.folds_num):
                Container = DataContainer(params, data, fold)
                xtr = Container.xtr
                ytr = Container.ytr
                xdev = Container.xdev
                ydev = Container.ydev

                # Create and fit the Lasso logistic regression model
                lasso_model = LogisticRegression(penalty='l1', C=c_value,
                                                 solver='liblinear')  # bigger c smaller regularization
                # lasso_model.fit(xtr, ytr)
                lasso_model.fit(xtr, ytr.squeeze())

                # Predict the model on the test set
                y_tr_pred = lasso_model.predict(xtr)
                train_acc.append(acc_score(ytr, y_tr_pred))
                y_dev_pred = lasso_model.predict(xdev)
                dev_acc.append(acc_score(ydev, y_dev_pred))
                print(f"train accuracy is:{str(train_acc[-1])}, and dev accuracy is:{str(dev_acc[-1])}")

            filename = params.res_directory + "LogisticRegression_c_value" + str(c_value) + ".mat"
            spio.savemat(filename, {'train_acc': train_acc, 'dev_acc': dev_acc})

    for hyper_hidden_dim in params.hyper_hidden_dims:
        for hidden_dim in params.hidden_dims:
            for learning_rate in params.learning_rates:
                for stg_regularizer in params.stg_regularizers:
                    start_time = time.time()  # Record start time
                    hyperparameter_combination = params.strfile + "_hidden" + str(hyper_hidden_dim) + "_lr" + \
                                                 str(learning_rate) + "_lam" + str(stg_regularizer)
                    print(hyperparameter_combination)
                    os.makedirs(os.path.join(params.res_directory, hyperparameter_combination), exist_ok=True)
                    acc_dev_folds = []
                    for fold in range(params.folds_num):
                        filename = os.path.join(params.res_directory, hyperparameter_combination,
                                                "selfold" + str(fold) + ".mat")

                        uneffective_flag = True
                        num_iter = 0
                        while uneffective_flag:
                            print(int(time.time()))
                            set_seed(int(time.time()))
                            # Data
                            Container = DataContainer(params, data, fold)
                            train_Dataloader, dev_Dataloader, test_Dataloader = Container.get_Dataloaders(params)

                            # Load model architecture
                            model = c_stg.models.__dict__[params.ML_model_name] \
                                (params.input_dim, hidden_dim, params.output_dim, params.param_dim, hyper_hidden_dim,
                                 params.dropout, sigma=params.sigma, include_B_in_input=params.include_B_in_input,
                                 non_param_stg=params.non_param_stg, train_sigma=params.train_sigma,
                                 classification=params.classification_flag)

                            model = model.to(params.device).float()
                            criterion = init_criterion(params)
                            optimizer = init_optimizer(model, learning_rate)

                            train_acc_array, train_loss_array, dev_acc_array, dev_loss_array, uneffective_flag = \
                                train(params, model, train_Dataloader, dev_Dataloader, criterion, optimizer,
                                      stg_regularizer, acc_score)

                            num_iter += 1
                            if num_iter == 20:
                                uneffective_flag=False

                        print("-----------------dev acc fold" + str(fold) + " is:" + str(dev_acc_array[-1]))
                        acc_dev_folds.append(dev_acc_array[-1])

                        unique_r = np.unique(Container.rte)
                        #alpha_vals = np.zeros((Container.xtr.shape[1], len(unique_r)))
                        mu_vals = np.zeros((Container.xtr.shape[1], len(unique_r)))
                        if params.ML_model_name == "fc_stg_layered_param_modular_model_sigmoid_extension":
                            w_vals = np.zeros((Container.xtr.shape[1], len(unique_r)))
                        else:
                            w_vals = []
                        acc_vals_per_r = np.zeros(len(unique_r))
                        ri = 0
                        for rval in np.unique(Container.rte):
                            #alpha_vals[:, ri] = get_prob_alpha(params, model, np.array(rval).reshape(-1, 1))
                            if params.ML_model_name == "fc_stg_layered_param_modular_model_sigmoid_extension":
                                mu_vals[:, ri], w_vals[:, ri] =\
                                    get_prob_alpha(params, model, np.array(rval).reshape(-1, 1))
                            elif params.ML_model_name == "fc_stg_layered_param_modular_model_sigmoid":
                                mu_vals[:, ri] = get_prob_alpha(params, model, np.array(rval).reshape(-1, 1))
                            inds = [i for i, x in enumerate(Container.rte == rval) if x]
                            x_test_tmp = Container.xte[inds, :]
                            r_test_tmp = Container.rte[inds].reshape(-1, 1)
                            y_test_tmp = Container.yte[inds].reshape(-1, 1)
                            # y_test_tmp = torch.empty_like(torch.tensor(r_test_tmp))
                            test_set_tmp = data_utils.TensorDataset(torch.tensor(x_test_tmp), torch.tensor(y_test_tmp),
                                                                    torch.tensor(r_test_tmp))
                            test_dataloader_tmp = torch.utils.data.DataLoader(test_set_tmp,
                                                                              batch_size=params.batch_size,
                                                                              shuffle=False)
                            acc_dev, _, _, _ = test_process(params, model, test_dataloader_tmp, criterion,
                                                            stg_regularizer, acc_score)
                            acc_vals_per_r[ri] = acc_dev
                            ri += 1

                        spio.savemat(filename,
                                     {'nn_acc_train': train_acc_array[-1], 'nn_acc_dev': dev_acc_array[-1],
                                      'train_acc_array': train_acc_array, 'dev_acc_array': dev_acc_array,
                                      'train_loss_array': train_loss_array, 'dev_loss_array': dev_loss_array,
                                      'unique_r': unique_r, 'mu_vals': mu_vals,
                                      'w_vals': w_vals, 'acc_vals_per_r': acc_vals_per_r})

                        end_time = time.time()  # Record end time
                        elapsed_time = end_time - start_time
                        # Convert elapsed time to hours, minutes, and seconds
                        hours, remainder = divmod(elapsed_time, 3600)
                        minutes, seconds = divmod(remainder, 60)

                    msg1 = f"\nTime taken for hyperparameters: " \
                           f"hyper_hidden_dim={hyper_hidden_dim}, hidden_dim={hidden_dim}, " \
                           f"learning_rate={learning_rate}, stg_regularizer={stg_regularizer}: " \
                           f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds\n"
                    msg2 = "-----------------Mean dev acc for taken hyperparameters:" + str(
                        np.array(acc_dev_folds).mean())
                    print(msg1, msg2)
                    with open(os.path.join(params.res_directory, 'log.txt'), 'a') as fe:
                        fe.write(msg1)
                        fe.write(msg2)

    print("----Start post-processing---")
    name = params.res_directory.split("\\")[-1]
    post_process_flow(data_type, name, cstg_args={}, data_args={})
    print("----FINISH----")



