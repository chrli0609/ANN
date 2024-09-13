import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from model import MLP
from functions import *

# Define x and y arrays
x = np.arange(-5, 5.5, 0.5)
y = np.arange(-5, 5.5, 0.5)

# Calculate the number of data points
ndata = len(x) * len(y)

# Create meshgrid for x and y
X, Y = np.meshgrid(x, y)

# Calculate Z
Z = np.exp(-X**2 * 0.1) * np.exp(-Y**2 * 0.1) - 0.5

# Reshape Z into a 1D array (targets)
targets = Z.reshape(1, ndata)

# Reshape X and Y into 1D arrays and concatenate them as patterns
patterns = np.vstack((X.reshape(1, ndata), Y.reshape(1, ndata)))

# Print shapes to verify
print("Targets shape:", targets.shape)
print("Patterns shape:", patterns.shape)

# Model parameters
in_dim, num_samples = patterns.shape
out_dim, _ = targets.shape
hidden_nodes = 15
LEARNING_RATE = 0.01
NUM_EPOCHS = 175

# Train the model
model = MLP(in_dim, hidden_nodes, out_dim, LEARNING_RATE)
model.training(patterns, targets, NUM_EPOCHS)

print("______________________________________________________________")

# Get predictions from the model
O, _ = model.forward_pass(patterns)

# Reshape the output to the grid size
Z_pred = np.reshape(O, (len(x), len(y)))

# Create a single figure with two 3D subplots
fig = plt.figure(figsize=(14, 7))

# Plot the original function
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('Original Function')
ax1.set_xlim([-5, 5])
ax1.set_ylim([-5, 5])
ax1.set_zlim([-0.7, 0.7])

# Plot the model's prediction
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z_pred, cmap='viridis')
ax2.set_title('Model Prediction')
ax2.set_xlim([-5, 5])
ax2.set_ylim([-5, 5])
ax2.set_zlim([-0.7, 0.7])

# Display the plots
plt.tight_layout()
plt.show()
