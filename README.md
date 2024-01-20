# neural_cstg
contextual feature selection for neural data

## Configuration Parameters

The `Data_params` class (located in `data_params.py`) should contain the following properties:

- **res_directory**: Directory path for saving results. Format: `../results/name_of_a_specific_sub_result_of_yours`.
- **folds_num**: Number of cross-validation folds.
- **manual_random_seed**: Seed for non-deterministic operations. Use `-1` for no specific seed setting.

## Data Processing

The `DataProcessor` class (located in `data_processing.py`) should contain the following properties:

- **params**: Initialize with the `params` object, and add the `chance_level` property. Should be like: `self.params = params; self.params.chance_level = calculated_chance_level` based on the data.
- **data_use**: Flag to determine data usage.
- **explan_feat**: Explanatory network features. Shape: `(trials*time x neurons)`.
- **output_label**: Output label. Shape: `(trials*time x 1)`.
- **context_feat**: Contextual network features. Shape: `(trials*time x 1)`.
- **traininds/testinds/devinds**: Lists containing `params.folds_num` sub-lists, each contain another list with train/test/dev indices for a cross-validation fold.

