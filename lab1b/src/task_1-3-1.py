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

# Prepare the figure and axes for the animation
fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Initialize the plot
def init():
    ax1.clear()
    ax2.clear()
    
    ax1.set_title('Original Function')
    ax1.set_xlim([-5, 5])
    ax1.set_ylim([-5, 5])
    ax1.set_zlim([-0.7, 0.7])
    
    ax2.set_title('Model Prediction')
    ax2.set_xlim([-5, 5])
    ax2.set_ylim([-5, 5])
    ax2.set_zlim([-0.7, 0.7])

# Update the plot for each frame in the animation
def update(frame):
    hidden_nodes = frame + 1  # Number of hidden nodes
    
    # Clear previous plots
    ax1.clear()
    ax2.clear()
    
    # Plot the original function
    ax1.plot_surface(X, Y, Z, cmap='viridis')
    ax1.set_title('Original Function')
    ax1.set_xlim([-5, 5])
    ax1.set_ylim([-5, 5])
    ax1.set_zlim([-0.7, 0.7])
    
    # Train the model
    model = MLP(IN_DIM=patterns.shape[0], NUM_IN_NODES=hidden_nodes, OUT_DIM=targets.shape[0], LEARNING_RATE=0.005)
    model.training(patterns, targets, NUM_EPOCHS=400)
    
    # Get predictions from the model
    O, _ = model.forward_pass(patterns)
    
    # Reshape the output to the grid size
    Z_pred = np.reshape(O, (len(x), len(y)))
    
    # Plot the model's prediction
    ax2.plot_surface(X, Y, Z_pred, cmap='viridis')
    ax2.set_title(f'Model Prediction with {hidden_nodes} Hidden Nodes')
    ax2.set_xlim([-5, 5])
    ax2.set_ylim([-5, 5])
    ax2.set_zlim([-0.7, 0.7])
    
    return ax1, ax2

# Create the animation
ani = FuncAnimation(fig, update, frames=25, init_func=init, blit=False, repeat=False, interval=1000)

# Save the animation as a GIF
ani.save('model_prediction_animation.gif', writer='imagemagick')

# Show the animation
plt.show()
