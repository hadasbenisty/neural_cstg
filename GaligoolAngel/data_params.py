
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
