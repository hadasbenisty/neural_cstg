from variables import feature_dim, output_dim, data_path, results_path, data_names, save_path
from define_nn import PredictionNetwork, CSTGModel
from training import training_k_fold
import torch
import scipy.io as io
from plot_functions import plot_weights, plot_model_results
from main import features, y
import numpy as np
# Loading the model
# Initializing the model
model = CSTGModel(feature_dim, output_dim)
model.load_state_dict(torch.load(save_path))

# Extracting the model
# Assuming 'model' is your trained PyTorch model
model_params = {}
for name, param in model.named_parameters():
    # Convert to numpy array
    model_params[name] = param.cpu().detach().numpy()

plot_model_results(model, features, y)
plot_weights(model.hypernetwork)

# Get model predictions
with torch.no_grad():
    y_pred = model(features).cpu().numpy()
y_np = y.numpy()
avg_error = np.mean(np.abs(y_pred - y_np)) / np.mean(y)
print(f'average error: {avg_error}%')