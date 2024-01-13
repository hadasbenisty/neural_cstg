import numpy as np

from variables import feature_dim, output_dim, data_path, results_path, data_names
from define_nn import PredictionNetwork, CSTGModel
from training import training_k_fold
import torch
import scipy.io as io
from visual_functions import calc_real_weights
from plot_functions import plot_weights

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# Adding the dataset
data_mat = io.loadmat(data_path)
features = (torch.from_numpy(data_mat[data_names['all']])).T
features = features.float()
feature_dim = len(features.T)

y = (torch.from_numpy(data_mat['y'])).T
y = y.float()

# Initializing the model
model = CSTGModel(feature_dim, output_dim)

# Moving to Device
model.to(device)
features.to(device)
y.to(device)

# Training the Model
model = training_k_fold(model, features, y)

# Extracting the model
# Assuming 'model' is your trained PyTorch model
model_params = {}
for name, param in model.named_parameters():
    # Convert to numpy array
    model_params[name] = param.cpu().detach().numpy()

print(sum(calc_real_weights((model.hypernetwork)) > 0))
print(np.average(calc_real_weights(model.hypernetwork)))
plot_weights(model.hypernetwork)
io.savemat(results_path, model_params)

# Forward pass
predictions = model(features)
