from c_stg import train_main


# main :
if __name__ == '__main__':
    our_folder = 'wesal_c_stg'

    cstg_params = {'ML_model_name': 'fc_stg_layered_param_modular_model_sigmoid',
                   'classification_flag': False, 'folds_num': 5, 'hidden_dims': [[4]],
                   'learning_rates': [0.001],
                   'num_epoch': 150, 'hyper_hidden_dims': [[10, 10]], 'stg_regularizers': [0.1, 1]}


    data_params = {'matfile_path': '../../data/inputs/dataset_fake.mat'}
    output = train_main.main_workflow(cstg_args=cstg_params, data_type=our_folder, data_args=data_params)



