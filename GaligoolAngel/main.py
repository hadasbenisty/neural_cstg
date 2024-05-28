
# from unittest.runner import _ResultClassType
from GaligoolAngel.data_analysis.group_result_analyzer import GroupResultAnalyzer
from GaligoolAngel.utils import vector_to_matrix_index, vector_to_symmetric_matrix
from c_stg.train_main import main_workflow
#from c_stg.train_main import main_workflow
from GaligoolAngel.data_analysis.result_processor import ResultProcessor
from GaligoolAngel.data_analysis.result_analyzer import ResultAnalyzer
import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
import os

# Calling the main workflow

our_folder = 'GaligoolAngel'
cstg_params = {'ML_model_name': 'fc_stg_layered_param_modular_model_sigmoid',
               'classification_flag': False, 'folds_num': 5, 'hidden_dims': [[200, 100], [900,90], [500, 120], [300, 100], [400, 200]],
               'learning_rates': [0.0005, 0.0005, 0.0001, 0.001],
               'num_epoch': 150, 'hyper_hidden_dims': [[30, 500] ,[50, 500],[30,500],[10,300],[100,1000],[100]], 'stg_regularizers': [0.1, 1]} # [50, 500],[30,500],[10,300],[100,1000],[100] stg 1
data_params = {'matfile_path': 'C:/Users/hadas-stud-group2/Documents/GitHub/neural_cstg/GaligoolAngel/data/inputs/dataset_diff_animal2_comp5.mat'}

# result_path = main_workflow(cstg_args=cstg_params, data_type=our_folder, data_args=data_params)

#result_path = '..\\results\\_2024_03_29_10_27_20_animal_1\\c-stg_hidden[30, 500]_lr0.0005_lam0.1_Final_check.mat'
# Analyzing Results
# result_path = '..\\results\\_2024_04_01_11_34_59_animal_1\\c-stg_hidden[30, 500]_lr0.0005_lam0.1_Final_check.mat'
# Analyzing Results
result_path = 'C://Users//hadas-stud-group2//OneDrive - Technion\First Degree//Project A//results//c-stg_hidden[50, 500]_lr0.0005_lam0.1_Final_check.mat'
data_params = {'matfile_path': 'C://Users//hadas-stud-group2//OneDrive - Technion//First Degree//Project A//data//inputs//graph_analysis_inputs.mat'}
results_processor = ResultProcessor(result_path)

# Importing Real Data
raw_data = spio.loadmat(data_params["matfile_path"])
cc = vector_to_symmetric_matrix(raw_data["features"])
session_partition = raw_data["context"]
sessions = np.unique(session_partition)
sessions_order = []
for session in sessions:
    _, tmp_order = np.where(session_partition == session)
    sessions_order.append(tmp_order)

# Taking Subset Of Features
cc_dynamic_analysis = cc[results_processor.dynamic_neurons[:, None, None], results_processor.dynamic_neurons[None, :, None], :].squeeze()
analyzer = ResultAnalyzer(cc_dynamic_analysis, results_processor.dynamic_neurons, sessions_order)
#analyzer.community_analysis()
analyzer.degree_analysis()
analyzer.eigen_values_analysis()
analyzer.avg_corr_analysis()
analyzer.plot_analysis_results("all", os.path.join(os.path.dirname(result_path), 'dynamic'))

cc_important_analysis = cc[results_processor.important_neurons[:, None, None], results_processor.important_neurons[None, :, None], :].squeeze()
analyzer_imp = ResultAnalyzer(cc_important_analysis, results_processor.important_neurons, sessions_order)
#analyzer.community_analysis()
analyzer_imp.degree_analysis()
analyzer_imp.eigen_values_analysis()
analyzer_imp.avg_corr_analysis()
analyzer_imp.plot_analysis_results("all", os.path.join(os.path.dirname(result_path), 'imp'))


## All Neurons
analyzer_all = ResultAnalyzer(cc, np.linspace(0, cc.shape[0]-1, cc.shape[0]).astype("int"), sessions_order)
analyzer_all.degree_analysis()
analyzer_all.eigen_values_analysis()
analyzer_all.avg_corr_analysis()
analyzer_all.plot_analysis_results("all", os.path.join(os.path.dirname(result_path), 'all'))

## Correlation Between SubSets:
subset_analyzer = GroupResultAnalyzer([analyzer_imp, analyzer, analyzer_all], ["important", "dynamic", "all"], cc, sessions_order)
subset_analyzer.corr_analysis(["important", "dynamic", "all"])
subset_analyzer.plot_results(os.path.join(os.path.dirname(result_path), 'subsets'), ["important", "dynamic", "all"])

## Save Results
save_path = os.path.join(os.path.dirname(result_path), 'post_analysis')
if not os.path.exists(save_path):
        os.makedirs(save_path)
analysis_results = {"analyzers": [analyzer, analyzer_imp, analyzer_all], 'subset_analyzer': subset_analyzer}
spio.savemat(os.path.join(save_path, 'analysis_results.mat'), analysis_results)
print("--- DONE ---")