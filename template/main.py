import os
from c_stg.train_main import main_workflow

if __name__ == '__main__':


    from template.utils import get_subdirectories
    data_directory = '..\\data'
    specific_data = "my_data
    print(f"WORKING ON ANIMAL:{specific_data}")
            data_dict = {'specific_data': specific_data}
            #cstg_dict = {'hyper_hidden_dims': 0}

            main_workflow(data_type='template', cstg_args={}, data_args=data_dict)
