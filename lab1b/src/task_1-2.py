import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from model import MLP
from functions import generate_random_non_linear_input_and_weights, subsampling_25_from_each_class

# Data and parameters
NUM_SAMPLES_PER_CLASS = 134
mA = np.array([1.0, 0.3])
sigmaA = 0.2
mB = np.array([0.0, -0.1])
sigmaB = 0.3

# Generate random data
IN_DIM = 2
NUM_EPOCHS = 20
LEARNING_RATE = 0.01
MAX_HIDDEN_NODES_TO_TRY = 50

# Generate data and weights
init_W, X, T, X_test, T_test, color_list = generate_random_non_linear_input_and_weights(
    IN_DIM, NUM_SAMPLES_PER_CLASS, mA, mB, sigmaA, sigmaB, subsampling_25_from_each_class
)

# Generate list of number of hidden nodes to try
hidden_nodes_list = np.arange(1, MAX_HIDDEN_NODES_TO_TRY + 1, 1)

# Arrays to store errors for each neuron configuration
train_error_hidden_nodes_list = []
valid_error_hidden_nodes_list = []

# Train and record errors for each neuron configuration
for hidden_nodes in hidden_nodes_list:
    model = MLP(IN_DIM, hidden_nodes, 1, LEARNING_RATE)
    train_error, valid_error = model.training_w_valid(X, T, X_test, T_test, NUM_EPOCHS)
    train_error_hidden_nodes_list.append(train_error)
    valid_error_hidden_nodes_list.append(valid_error)

# Convert errors to arrays for easy indexing
train_error_hidden_nodes_list = np.array(train_error_hidden_nodes_list)
valid_error_hidden_nodes_list = np.array(valid_error_hidden_nodes_list)

# Animation function
def animate_training_validation_errors():
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Initialize lines for training and validation error
    train_line, = ax.plot([], [], label='Training Error', color='blue')
    val_line, = ax.plot([], [], label='Validation Error', color='red')

    # Set axis limits
    ax.set_xlim(1, NUM_EPOCHS)
    ax.set_ylim(0, 2)  # Adjust based on your error scale
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error')
    ax.set_title('Training and Validation Errors Across Epochs')
    ax.legend()

    # Add grid
    ax.grid(True)

    # Set x and y ticks
    ax.set_xticks(np.arange(1, NUM_EPOCHS + 1, 1))
    ax.set_yticks(np.linspace(0, 2.1, num=22))

    # Create text annotation for neuron count
    neuron_text = ax.text(0.8, 0.9, '', transform=ax.transAxes, fontsize=12, color='black',
                          bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    # Function to initialize the plot
    def init():
        train_line.set_data([], [])
        val_line.set_data([], [])
        neuron_text.set_text('')
        return train_line, val_line, neuron_text

    # Function to update the plot for each frame in the animation
    def update(frame):
        neurons = hidden_nodes_list[frame]  # Current number of neurons

        # Update training and validation lines with the corresponding errors
        train_line.set_data(np.arange(1, NUM_EPOCHS + 1), train_error_hidden_nodes_list[frame])
        val_line.set_data(np.arange(1, NUM_EPOCHS + 1), valid_error_hidden_nodes_list[frame])

        # Update the title to reflect the current neuron count
        ax.set_title(f'Training and Validation Errors for {neurons} Neurons')

        # Update the neuron count text annotation
        neuron_text.set_text(f'Neurons: {neurons}')

        return train_line, val_line, neuron_text

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(hidden_nodes_list), init_func=init, blit=True, interval=600)

    # Show the animation
    plt.show()

# Call the animation function when needed
animate_training_validation_errors()

