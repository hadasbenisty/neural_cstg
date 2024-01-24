from c_stg import train_main

# Calling the main workflow
our_folder = 'GaligoolAngel'
cstg_params = {'ML_model_name': 'fc_stg_layered_param_modular_model_sigmoid'}
output = train_main.main_workflow(cstg_args=cstg_params, data_type=our_folder)
