
import sys
import os
import paths  # TODO Fix that we add paths from another table
from variables import feature_dim, output_dim, data_names
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

# Adding the dataset
data_mat = io.loadmat(paths.input_paths)
features = (torch.from_numpy(data_mat[data_names['all']])).T
features = features.float()
feature_dim = len(features.T)

y = (torch.from_numpy(data_mat['y'])).T
y = y.float()
output_dim = len(y.T)

# Initializing the model
model = CSTGModel(feature_dim, output_dim)

# Moving to Device
model.to(device)
features.to(device)
y.to(device)

# Training the Model
model = training_k_fold(model, features, y, 3)

# Forward pass
predictions = model(features)

# Save the model
torch.save(model.state_dict(), os.path.join(paths.results_paths, "model.pth"))

# Save Predictions to use in matlab
predictions_np = predictions.detach().numpy()
model_weights = model.state_dict()
# Convert the weights to NumPy arrays
weights_np = {k: v.cpu().numpy() for k, v in model_weights.items()}
# Replace periods in keys with underscores
weights_np = {k.replace('.', '_'): v.cpu().numpy() for k, v in model_weights.items()}

results = {"predictions": predictions_np, "model_weights": weights_np}
io.savemat(os.path.join(paths.results_paths, "results.mat"), results)

