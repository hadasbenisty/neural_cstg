import os.path
from datetime import datetime
import numpy as np
import scipy.io as spio

class Data_params(object):
    def __init__(self, **kwargs):
        ######################
        # Data Parameters #
        ######################
        # -----------------------------------
        self.mat_files_directory = '..\\data\\'
        self.result_directory = '..\\results\\'
        self.matfile_path = '..\\data\\inputs\\dataset.mat'
        self.folds_num = 5  # cross validation
        self.result_name = ''
        # Extra parameters
        self.start_time = -4  # sec
        self.end_time = 8  # sec
        self.animal = '1'

        self.manual_random_seed = -1


        for key, value in kwargs.items():
            setattr(self, key, value)





def data_origanization_params(params):
    running_datetime = datetime.now()  # TODO: Fix date
    running_datetime = running_datetime.strftime("%Y_%m_%d_%H_%M_%S")
    sub_res_directory = '_' + running_datetime + '_animal_' + params.animal
    sub_res_directory.replace(':', '_').replace(' ', '_')

    params.res_directory = params.result_directory + sub_res_directory
    params.mat_files_directory = params.mat_files_directory + params.animal
    # params.sheet_num = params.animal2sheet_num[params.animal]

    # params.bar_graph = True if params.context_key == 'flavors' else False
    # Calculate Inner Results Directory
    data = spio.loadmat(params.matfile_path)
    if 'fake' in params.matfile_path:
        number_features = data["features"].shape[0]
        number_outputs = data["y"].shape[0]
        number_context = np.max(np.unique(data["context"]))
        number_meas = data["context"].shape[1]
        SNR_level = int(data["SNR"].flatten())
        folder_name = (f"number_features_{number_features}_number_outputs_"
                       f"{number_outputs}_number_context_{number_context}_number_meas_"
                       f"{number_meas}_SNR_level_{SNR_level}")
        params.res_directory = '_'.join([params.res_directory, folder_name])
    return params
