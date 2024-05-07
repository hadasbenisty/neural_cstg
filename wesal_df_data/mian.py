import os

import pandas as pd

from c_stg import train_main

if __name__ == '__main__':
    our_folder = 'wesal_df_data'
    cstg_params = {'ML_model_name': 'fc_stg_layered_param_modular_model_sigmoid',
                   'folds_num': 5, 'hidden_dims': [[900, 90], [500, 120], [200, 100]],
                   'learning_rates': [0.0005, ],
                   'num_epoch': 10, 'hyper_hidden_dims': [[50, 500], [30, 500], [10, 300], [100, 1000], [100]],
                   'stg_regularizers': [0.1],
                   'include_linear_model': True}

    df = pd.read_csv(r"C:\Users\WesalAwida\PycharmProjects\neural_cstg\wesal_df_data\suicide_data\suicide_data.csv")

    # Retrieve column names
    columns = df.columns
    print(columns)
    x = ['Age', 'Sex', 'Financial_problems', 'Two_Parents_Household',
         'Adoptionor_FosterCare', 'Childrens_Aid_Service',
         'Family_Relationship_Difficulties', 'Between_Caregivers_Violence',
         'Caregiver_To_Child_Violence', 'Head_Injury', 'Stimulant_Meds',
         'Full_Scale_IQ', 'WISC_Vocabulary', 'WISC_BlockDesign',
         'Social_Withdrawal', 'Social_Conflicts', 'Academic_Difficulty',
         'School_Truancy', 'Inattention', 'Hyperactivity_Impulsivity',
         'Irritability', 'Defiance', 'Aggresive_Conduct_Problems',
         'NonAggresive_Conduct_Problems', 'Depression', 'Anxiety',
         'Sleep_Prolems', 'Somatization']

    y = ['Parent_Reported_Suicidality', 'Parent_Reported_SI', 'Parent_Reported_SB', 'Self_Reported_Sl']

    for i, context_feat in enumerate(x):
        for j, target in enumerate(y):
            print(f"Processing for target variable: {target}")
            cstg_params["context_feat"] = context_feat
            cstg_params["target"] = target

            output = train_main.main_workflow(cstg_args=cstg_params, data_type=our_folder, data_args={})
