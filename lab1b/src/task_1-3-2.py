import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
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
LEARNING_RATE = 0.005
NUM_EPOCHS = 175
hidden_nodes = 18  # Adjust as needed

# Prepare for animation
fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Plot the original function
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('Original Function')
ax1.set_xlim([-5, 5])
ax1.set_ylim([-5, 5])
ax1.set_zlim([-0.7, 0.7])

# Define update function for animation
def update(frame):
    nsamp = frame + 1
    perm = np.random.permutation(num_samples)
    selected_patterns = patterns[:, perm[:nsamp]]
    selected_targets = targets[:, perm[:nsamp]]

    model = MLP(in_dim, hidden_nodes, out_dim, LEARNING_RATE)
    model.training(selected_patterns, selected_targets, NUM_EPOCHS)

    # Get predictions from the model
    O, _ = model.forward_pass(patterns)
    
    # Clear the previous plot
    ax2.clear()
    
    # Reshape the output to the grid size
    Z_pred = np.reshape(O, (len(x), len(y)))

    # Plot the model's prediction
    ax2.plot_surface(X, Y, Z_pred, cmap='viridis')

    # Set axis limits
    ax2.set_xlim([-5, 5])
    ax2.set_ylim([-5, 5])
    ax2.set_zlim([-0.7, 0.7])
    ax2.set_title(f'Model with {nsamp} Samples of 441')
    
    return ax2,

# Create animation
ani = FuncAnimation(fig, update, frames=441, interval=20, blit=False)

# Save animation as GIF
ani.save('model_training_animation.gif', writer='pillow')

# Show plot
plt.show()
