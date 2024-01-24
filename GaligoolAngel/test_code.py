from data_processing import DataProcessor
from c_stg.params import Params_config
import os
cstg_params = {}
our_dir = 'GaligoolAngel'
params = Params_config(our_dir)
print(os.getcwd())
data = DataProcessor(params, '..\\..\\data\\dataset.mat')
print('hi')