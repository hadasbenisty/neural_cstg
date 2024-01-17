import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_weights(model, layer_number=1):
    weights = model.weights.detach().numpy()
    # Use imshow to plot the weight matrix
    # Create a range of values for the x-axis
    x_values = np.arange(len(weights))
    # Create the bar heatmap
    plt.bar(x_values, weights, color='skyblue')
    # plt.colorbar()  # Adds a color bar to interpret the values
    plt.title('Weights Matrix Visualization')
    plt.xlabel('Input Features')
    plt.ylabel('Neurons in the Layer')
    plt.show()


def plot_model_results(model, X, y_true):
    """
    Plots the model's predictions against the true values.

    :param model: Trained PyTorch neural network model.
    :param X: Input features (torch.Tensor).
    :param y_true: True output values (torch.Tensor).
    """
    model.eval()  # Set the model to evaluation mode

    # Get model predictions
    with torch.no_grad():
        y_pred = model(X).cpu().numpy()

    # Check if the output is 1D or 2D, raise an error if greater than 2D
    if y_pred.ndim > 2 or (y_pred.ndim == 2 and y_pred.shape[1] > 2):
        raise ValueError("Function only supports output dimensions of 1D or 2D.")

    # Number of measurements
    num_measurements = len(y_pred)

    # Plotting
    if y_pred.ndim == 1 or y_pred.shape[1] == 1:  # 1D output
        plt.figure(figsize=(10, 6))
        plt.plot(range(num_measurements), y_true.cpu().numpy(), label='True Values', marker='o')
        plt.plot(range(num_measurements), y_pred, label='Predictions', marker='x')
        plt.xlabel('Measurement Number')
        plt.ylabel('Output Value')
        plt.legend()
    else:  # 2D output
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        axs[0].plot(range(num_measurements), y_true.cpu().numpy()[:, 0], label='True Values - Dim 1', marker='o')
        axs[0].plot(range(num_measurements), y_pred[:, 0], label='Predictions - Dim 1', marker='x')
        axs[1].plot(range(num_measurements), y_true.cpu().numpy()[:, 1], label='True Values - Dim 2', marker='o')
        axs[1].plot(range(num_measurements), y_pred[:, 1], label='Predictions - Dim 2', marker='x')
        for ax in axs:
            ax.set_xlabel('Measurement Number')
            ax.set_ylabel('Output Value')
            ax.legend()

    plt.suptitle('Model Output vs. True Values')
    plt.show()

# Example usage
# model = YourTrainedModel()
# plot_model_results(model, X_test_tensor, y_test_tensor)

