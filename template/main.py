import os
from c_stg.train_main import main_workflow

if __name__ == '__main__':
    # from flavors.utils import get_subdirectories
    # data_directory = '..\\data'
    # for animal in os.listdir(data_directory):
    #     print(animal)
    #     if animal != '4575' and animal != 'animals_db_selected.xlsx' and animal != '4458' and animal != '4754' and animal != '4756':
    #         animal_directory = data_directory + "\\" + animal
    #         dates = get_subdirectories(animal_directory)
    #         if animal == '4754':
    #             dates = dates[1:]
    #         for date in dates:
    #             print(f"WORKING ON ANIMAL:{animal},DATE:{date}")
    #             arguments_dict = {'animal': animal, 'date': date}
    #             main_workflow(**arguments_dict)

    from flavors.utils import get_subdirectories
    data_directory = '..\\data'
    #animals = ['1111', '4458']
    animals = ['1111']
    for animal in animals:
        animal_directory = data_directory + "\\" + animal
        if animal == '4458':
            dates = get_subdirectories(animal_directory)
            dates = [dates[2]]
        elif animal == '1111':
            dates = ['01_20_24']
        for date in dates:
            print(f"WORKING ON ANIMAL:{animal}, DATE:{date}")
            data_dict = {'animal': animal, 'date': date}
            #cstg_dict = {'hyper_hidden_dims': 0}

            main_workflow(data_type='flavors', cstg_args={}, data_args=data_dict)