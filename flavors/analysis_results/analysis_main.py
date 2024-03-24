# Imports
from result_analyser import ResultsAnalyser


animals = ['4458_0', '4575_1', '4754_2', '4756_3', '4880_4', '4882_5']
main_parts = ['first', 'ongoing_batch', 'ongoing_random']
animal2sheet_num = {'4458': 0, '4575': 1, '4754': 2,
                    '4756': 3, '4880': 4, '4882': 5, '4940': 6, '1111': 7}
info_excel_path = '../../data/animals_db_selected.xlsx'
res_directory = '../../results'
num_folds = 5
flavors2num = {'g': 1, 's': 2, 'q': 3, 'f': 4}  # treat 'r' as equal to 'g'

context = 'flavors'
classification_type = 'success'
resultsAnalyser = ResultsAnalyser(res_directory, context, classification_type, animals, main_parts, animal2sheet_num,
                 info_excel_path, num_folds, flavors2num)
# chosen_directory = '2024_01_01_20_48_09_animal_4575_date_03_19_19_success'
# resultsAnalyser.open_gates_visual(chosen_directory, min_th=0.2, max_th=0.8, num_neu=5)
# #
#resultsAnalyser.acc_vs_time_per_parts()
#esultsAnalyser.open_gate_percents_vs_time_per_parts()
#
# resultsAnalyser.classification_type = 'flavors'
# resultsAnalyser.extract_run_type()
# #resultsAnalyser.acc_vs_time_per_parts(num_flavors_relevant=2)
# resultsAnalyser.acc_vs_time_per_parts(num_flavors_relevant=3)

# resultsAnalyser.open_gate_percents_vs_time_per_parts()

# resultsAnalyser.compare_percent_open_gates()

# resultsAnalyser.classification_type = 'flavors'
# resultsAnalyser.extract_run_type()
# resultsAnalyser.conf_mat_at_specific_time_per_part(time_point=-1.5)
# resultsAnalyser.conf_mat_at_specific_time_per_part(time_point=2)
# resultsAnalyser.conf_mat_at_specific_time_per_part(time_point=7)

# resultsAnalyser.classification_type = 'success'
# resultsAnalyser.corr_parts_time_cotext_all_animals()

resultsAnalyser.corr_parts_flavors_context_all_animals()
#resultsAnalyser.analysis_per_flavor_per_parts_vs_time_windows('open_gates', num_flavors_relevant=3) #todo:maybe do for both 2 and 3 numbers
#resultsAnalyser.analysis_per_flavor_per_parts_vs_time_windows('acc_per_flavors', num_flavors_relevant=3)

print('end')