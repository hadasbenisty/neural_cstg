from c_stg import train_main
#from c_stg.train_main import main_workflow


# Calling the main workflow
our_folder = 'GaligoolAngel'
cstg_params = {'ML_model_name': 'fc_stg_layered_param_modular_model_sigmoid',
               'classification_flag': False, 'folds_num': 5, 'hidden_dims': [[900,90], [500, 120],[200,100]],
               'learning_rates': [0.0005, 0.0001, 0.001],
               'num_epoch': 150, 'hyper_hidden_dims': [[50, 500],[30,500],[10,300],[100,1000],[100]], 'stg_regularizers': [0.1, 1]}
data_params = {'matfile_path': 'C:/Users/hadas-stud-group2/Documents/GitHub/neural_cstg/GaligoolAngel/data/inputs/dataset_diff_animal2.mat'}
output = train_main.main_workflow(cstg_args=cstg_params, data_type=our_folder, data_args=data_params)

