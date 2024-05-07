from datetime import datetime


class Data_params(object):
    def __init__(self, **kwargs):
        """
        Data parameters class
        """
        # required:

        self.manual_random_seed = -1  # seed for not deterministic operations
        self.result_directory = '../results/'
        self.folds_num = 5  # cross validation
        self.mat_files_directory = '../wesal_df_data/suicide_data/'
        self.matfile_path = self.mat_files_directory + 'suicide_data.csv'
        # shira :
        self.start_time = -4  # sec
        self.end_time = 8  # sec

        for key, value in kwargs.items():
            setattr(self, key, value)


def data_origanization_params(params):
    """
    This function creates the result directory , the main folder
    """""
    # todo : @Wesal Add the post pressing mode
    # todo : @Wesal change it to somthing more informative
    running_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    sub_res_directory = running_datetime.replace(':', '_').replace(' ', '_')

    params.res_directory = params.result_directory + sub_res_directory
    return params
