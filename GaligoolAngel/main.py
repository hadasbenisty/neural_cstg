import numpy as np

from variables import feature_dim, output_dim, data_path, results_path
from define_nn import PredictionNetwork, CSTGModel
from training import training_k_fold
import torch
import scipy.io as io

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")


# Initializing the model
model = CSTGModel(feature_dim, output_dim)

# Adding the dataset
data_mat = io.loadmat(data_path)
features = (torch.from_numpy(data_mat['CC_features'])).T
features = features.float()
y = (torch.from_numpy(data_mat['diffusion_map'])).T
y = y.float()

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

print(sum(model_params['hypernetwork.weights'] > 0))
print(np.average(model_params['hypernetwork.weights']))
io.savemat(results_path, model_params)

# Forward pass
predictions = model(features)
