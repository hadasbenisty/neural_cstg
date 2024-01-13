import matplotlib.pyplot as plt
import numpy as np


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

