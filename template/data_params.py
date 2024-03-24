from datetime import datetime


class Data_params(object):
    def __init__(self, **kwargs):
        ######################
        # Data Parameters #
        ######################
        # ---- main user input ----
        #self.(specific fields for the data) = ...
        # define label for prediction                
        self.outcome_keys = ['success']  # replace with you own label
        # define label for context
        self.context_key = 'time' #replace time with your own context
        // add your personal documentation
        self.note = "new model, 4458 with flavors context and success classification, 4-8 time, with 40 times same acc until run again"  # save in the log context for self use notes
        # ---- post processing user input ----
        self.manual_random_seed = -1  # seed for not determensric operations
        # -----------------------------------
        self.mat_files_directory = '../data/'  # replace with you own 
        self.result_directory = '../results/'  # replace with you own 
        self.info_excel_path = '../data/animals_db_selected.xlsx'  # replace with you own if relevant
        self.outcome_options = ['success', 'fake', 'grain', 'sucrose', 'quinine', 'flavors']  # replace with you own 
        self.context_options = ['context1', 'context2'] # replace with your context labels
        
        self.folds_num = 5  # cross validation  replace with you own 
       

        for key, value in kwargs.items():
            setattr(self, key, value)


def data_origanization_params(params):
    if params.post_process_mode:
        # example code for the way output folder is extracted for flavors project. do this in a later stage
        
        # animal_idx = params.infer_directory.find('animal')
        # animal_idx = animal_idx + len('animal') + 1
        # params.animal = params.infer_directory[animal_idx:].split('_')[0]

        # date_idx = params.infer_directory.find('date')
        # date_idx = date_idx + len('date') + 1
        # params.date = '_'.join(params.infer_directory[date_idx:].split('_')[0:3])

        # outcome_keys_idx = params.infer_directory.find(params.date)
        # outcome_keys_idx = outcome_keys_idx + len(params.date) + 1
        # params.outcome_keys = (params.infer_directory[outcome_keys_idx:]).split('_')

    else:
        # example code for the way output folder is determined for flavors project. change this according to the specific experiment 
        running_datetime = datetime.now()
        running_datetime = running_datetime.strftime("%Y_%m_%d_%H_%M_%S")
        sub_res_directory = running_datetime + '_animal_' + params.animal + '_date_' + params.date + '_' + '_'.join(params.outcome_keys)
        sub_res_directory.replace(':', '_').replace(' ', '_')

        params.res_directory = params.result_directory + sub_res_directory  # '/home/shiralif/results/'
    params.mat_files_directory = params.mat_files_directory + params.animal
    params.sheet_num = params.animal2sheet_num[params.animal]

    # params.bar_graph = True if params.context_key == 'flavors' else False

    return params
