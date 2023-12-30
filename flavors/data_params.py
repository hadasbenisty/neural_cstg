from datetime import datetime


class Data_params(object):
    def __init__(self):
        ######################
        # Data Parameters #
        ######################
        # ---- main user input ----
        self.animal = '4458'  # mouse number
        self.date = '02_24_19'  # date of experiment
        self.outcome_keys = ['success']  # all the outcomes to consider
        # ---- post processing user input ----
        self.manual_random_seed = -1  # seed for not determensric operation
        # -----------------------------------
        self.mat_files_directory = 'D:\\flavorsProject\data\\'  # '/home/shiralif/data/'
        self.result_directory = '..\\results\\'  # '/home/shiralif/results/'
        self.info_excel_path = '..\\data\\animals_db_selected.xlsx'  # '/home/shiralif/data/animals_db_selected.xlsx'
        self.outcome_options = ['success', 'fake', 'regular', 'sucrose', 'quinine', 'tone']  # labels options
        self.animal2sheet_num = {'4458': 0, '4575': 1, '4754': 2,
                                 '4756': 3, '4880': 4, '4882': 5, '4940': 6}  # excel sheet num
        self.folds_num = 5  # cross validation
        self.start_time = -4  # sec
        self.end_time = 8  # sec
        self.drop_time = 1  # drop the first sec
        self.sample_per_sec = 30  # samples/sec
        self.window_size_avg = 1  # sec
        self.overlap_avg = 0.5  # sec


def data_origanization_params(params):
    if params.post_process_mode:

        animal_idx = params.infer_directory.find('animal')
        animal_idx = animal_idx + len('animal') + 1
        params.animal = params.infer_directory[animal_idx:].split('_')[0]

        date_idx = params.infer_directory.find('date')
        date_idx = date_idx + len('date') + 1
        params.date = '_'.join(params.infer_directory[date_idx:].split('_')[0:3])

        outcome_keys_idx = params.infer_directory.find(params.date)
        outcome_keys_idx = outcome_keys_idx + len(params.date) + 1
        params.outcome_keys = (params.infer_directory[outcome_keys_idx:]).split('_')

    else:
        running_datetime = datetime.now()
        running_datetime = running_datetime.strftime("%Y_%m_%d_%H_%M_%S")
        sub_res_directory = running_datetime + '_animal_' + params.animal + '_date_' + params.date + '_' + '_'.join(
            params.outcome_keys)
        sub_res_directory.replace(':', '_').replace(' ', '_')

        params.res_directory = params.result_directory + sub_res_directory  # '/home/shiralif/results/'
    params.mat_files_directory = params.mat_files_directory + params.animal
    params.sheet_num = params.animal2sheet_num[params.animal]

    return params
