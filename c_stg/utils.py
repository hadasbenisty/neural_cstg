import importlib
import numpy as np

def import_per_data_type(data_type):
    # Specific data imports
    # utils
    utils_module = importlib.import_module(f'{data_type}.utils')
    acc_score = getattr(utils_module, 'acc_score')
    set_seed = getattr(utils_module, 'set_seed')
    init_criterion = getattr(utils_module, 'init_criterion')
    init_optimizer = getattr(utils_module, 'init_optimizer')
    hyperparameters_chosen_extraction = getattr(utils_module, 'hyperparameters_chosen_extraction')
    # data processing
    data_processing = importlib.import_module(f'{data_type}.data_processing')
    DataProcessor = getattr(data_processing, 'DataProcessor')
    # data params
    data_params = importlib.import_module(f'{data_type}.data_params')
    data_origanization_params = getattr(data_params, 'data_origanization_params')
    Data_params = getattr(data_params, 'Data_params')
    # visual
    visual = importlib.import_module(f'{data_type}.visual')
    post_process_visualization = getattr(visual, 'visual_results')

    return (acc_score, set_seed, init_criterion, init_optimizer, DataProcessor, data_origanization_params, Data_params,
            hyperparameters_chosen_extraction, post_process_visualization)


def norm_minmax(x):
    if len(np.unique(x)) == 1:
        return x
    m = np.min(x)
    M = np.max(x)
    x = (x - m) / (M - m)
    return x