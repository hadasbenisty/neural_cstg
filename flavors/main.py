import os
from c_stg.train_main import main_workflow

if __name__ == '__main__':
    from flavors.utils import get_subdirectories
    data_directory = '..\\data'
    animals = ['4575', '4754', '4756', '4880', '4882', '4458']
    for animal in animals:
        animal_directory = data_directory + "\\" + animal
        if animal == '4458':
            subdirs = get_subdirectories(animal_directory)
            subdirs = subdirs[2:]
        elif animal == '4575':
            subdirs = get_subdirectories(animal_directory)
            subdirs = subdirs[2:]
        elif animal == '4754':
            subdirs = get_subdirectories(animal_directory)
            subdirs = subdirs[2:]
        elif animal == '4756':
            subdirs = get_subdirectories(animal_directory)
            subdirs = subdirs[1:]
        elif animal == '4880':
            subdirs = get_subdirectories(animal_directory)
            subdirs = subdirs[3:]
        elif animal == '4882':
            subdirs = get_subdirectories(animal_directory)
            subdirs = subdirs[2:]

        for subdir in subdirs:
            full_subdir = os.path.join(animal_directory, subdir)
            date = subdir
            print(f"WORKING ON ANIMAL:{animal}, DATE:{date}")
            data_dict = {'animal': animal, 'date': date}
            main_workflow(data_type='flavors', cstg_args={}, data_args=data_dict)
