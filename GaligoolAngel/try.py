import numpy as np
from scipy.io import savemat
from variables import results_path

test_data = {'array1': np.array([1, 2, 3]), 'array2': np.array([4, 5, 6])}
print(test_data)
savemat(results_path, test_data)
