import torch
from flavors.data_params import Data_params


class Cstg_params(object):
    def __init__(self):
        ######################
        # Model Parameters #
        ######################
        self.hyper_hidden_dims = [[10], [50], [100], [1000]]  # units for the gates
        self.hidden_dims = [[500, 300, 100, 50, 10, 2]]
        self.learning_rates = [0.0005, 0.001, 0.05]
        self.stg_regularizers = [0.05]  # this is lambda 0.0001,0.0005, 0.001
        self.dropout = 0
        self.train_sigma = False
        self.sigma = 0.5
        self.inverse_regularization = [0.01, 0.1, 1, 10, 100]
        ######################
        # Running Parameters #
        ######################
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ML_model_name = "fc_stg_layered_param_modular_model_sigmoid"
        self.include_linear_model = False
        self.post_process_mode = False  # after findig hyper-parameters
        self.num_epoch = 1
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
        # stg,include_B_in_input,non_param_stg = (False,False,False)
        # # 1b. no gates - with contextual information feeding into classifier
        # stg,include_B_in_input,non_param_stg = (False,True,False)


class Params(Cstg_params, Data_params):
    def __init__(self):
        Data_params.__init__(self)
        Cstg_params.__init__(self)
