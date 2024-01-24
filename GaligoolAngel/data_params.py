import datetime


class Data_params(object):
    def __init__(self, **kwargs):
        ######################
        # Data Parameters #
        ######################
        # -----------------------------------
        self.mat_files_directory = '../data/'
        self.result_directory = '../results/'
        self.folds_num = 5  # cross validation

        # Extra parameters
        self.start_time = -4  # sec
        self.end_time = 8  # sec
        self.animal_num = 1

        for key, value in kwargs.items():
            setattr(self, key, value)


def data_origanization_params(params):
    running_datetime = datetime.now()
    running_datetime = running_datetime.strftime("%Y_%m_%d_%H_%M_%S")
    sub_res_directory = running_datetime + '_animal_' + params.animal + \
                        '_date_' + params.date + '_' + '_'.join(params.outcome_keys)
    sub_res_directory.replace(':', '_').replace(' ', '_')

    params.res_directory = params.result_directory + sub_res_directory
    params.mat_files_directory = params.mat_files_directory + params.animal
    params.sheet_num = params.animal2sheet_num[params.animal]

    # params.bar_graph = True if params.context_key == 'flavors' else False

    return params
