import torch
import importlib

class Cstg_params(object):
    def __init__(self,  **kwargs):
        ######################
        # Model Parameters #
        ######################
        self.hyper_hidden_dims = [[10], [50], [100], [1000]]  # units for the gates
        # todo: @wesal
        self.hidden_dims = [[500, 300, 100, 50, 10, 2], [100, 50, 10, 2]]
        self.learning_rates = [0.0005, 0.001, 0.05]
        self.stg_regularizers = [0.005, 0.05, 0.1, 0.5]  #[0.05]  # 0.005, 0.0005  # this is lambda
        self.dropout = 0
        self.train_sigma = False
        self.sigma = 0.5
        #todo : @wesal check with shira :
        self.inverse_regularization = [0.01, 0.1, 1, 10, 100]
        ######################
        # Running Parameters #
        ######################
        self.classification_flag = True  # relevant for model init
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ML_model_name = "fc_stg_layered_param_modular_model_sigmoid"#"fc_stg_layered_param_modular_model_sigmoid_extension"
        self.include_linear_model = False
        self.post_process_mode = False  # after findig hyper-parameters
        self.num_epoch = 50  # Todo
        self.batch_size = 32
        # 3. parametric stg
        self.stg, self.include_B_in_input, self.non_param_stg = (True, False, False)
        self.strfile = 'c-stg'
        # # 2. stg - with contextual information NOT feeding into classifier
        # stg,include_B_in_input,non_param_stg = (True,False,True)
        # strfile = 'stg'
        # # 2b. stg - with contextual information feeding into classifier
        # stg,include_B_in_input,non_param_stg = (True,True,True)
        # # 1. no gates
        # stg,include_B_in_input,non_param_stg = (False,False,False) #@wesal without context
        # # 1b. no gates - with contextual information feeding into classifier
        # stg,include_B_in_input,non_param_stg = (False,True,False)

        for key, value in kwargs.items():
            setattr(self, key, value)


def Params_config(data_type,  cstg_kwargs={}, data_kwargs={}):
    data_params = importlib.import_module(f'{data_type}.data_params')
    Data_params = getattr(data_params, 'Data_params')

    class Params_config_inner(Cstg_params, Data_params):
        def __init__(self, cstg_kwargs, data_kwargs):

            Cstg_params.__init__(self, **cstg_kwargs)
            Data_params.__init__(self, **data_kwargs)

    return Params_config_inner(cstg_kwargs, data_kwargs)

