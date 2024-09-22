import numpy as np
import math
from rbf import RBF
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the sine function
def sine_func(x):
    return np.sin(2 * x)

# Define the square function
def square_func(x):
    return 1 if sine_func(x) > 0 else -1

# Function to generate training and testing data
def generate_data(x_end, step_length, function, noise):
    train_X = np.arange(0, x_end, step_length).reshape(-1, 1)
    test_X = np.arange(0.05, x_end, step_length).reshape(-1, 1)

    N_train_samples = len(train_X)
    N_test_samples = len(test_X)

    train_F = np.zeros((N_train_samples, 1))
    test_F = np.zeros((N_test_samples, 1))

    for i in range(len(train_X)):
        train_F[i] = function(train_X[i])

    for i in range(len(test_X)):
        test_F[i] = function(test_X[i])

    if noise:
        train_F += np.random.normal(0, 0.1, size=(N_train_samples, 1))
        test_F += np.random.normal(0, 0.1, size=(N_test_samples, 1))

    return train_X, train_F, test_X, test_F

# Function to update the plots for the animation
def update_plot(num, ax1, ax2, sine_test_X, sine_test_F, sine_pred_F_units, sine_pred_F_var):
    ax1.clear()
    ax2.clear()

    # Constant variance for RBF unit variation
    const_variance = 0.1
    # Constant number of RBF units for variance variation
    const_rbf_units = 15

    # Plot for varying RBF units (ax1)
    pred_units = sine_pred_F_units[num]
    abs_residual_units = np.abs(sine_test_F - pred_units)
    mean_abs_error_units = np.mean(abs_residual_units)
    
    ax1.plot(sine_test_X, sine_test_F, label="True", color="blue")
    ax1.plot(sine_test_X, pred_units, label="Pred", color="red")
    ax1.set_title(f"Predicted function with varying number of RBF units", fontsize=10)
    ax1.text(0.8, 0.95, f'RBF Units: {num + 5}\n$\sigma^2$: {const_variance}\nErr: {mean_abs_error_units:.4f}', transform=ax1.transAxes, 
               fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    ax1.legend()

    # Plot for varying variance (ax2)
    pred_var = sine_pred_F_var[num]
    abs_residual_var = np.abs(sine_test_F - pred_var)
    mean_abs_error_var = np.mean(abs_residual_var)
    
    ax2.plot(sine_test_X, sine_test_F, label="True", color="blue")
    ax2.plot(sine_test_X, pred_var, label="Pred", color="red")
    ax2.set_title(f"Predicted function with varying variance", fontsize=10)
    ax2.text(0.8, 0.95, f'RBF units: {const_rbf_units}\n$\sigma^2$: {0.02 * (num + 1):.2f}\nErr: {mean_abs_error_var:.4f}', transform=ax2.transAxes, 
               fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    ax2.legend()

# Main code for setting up and animating
STEP_LENGTH = 0.1
NOISE = False
LEARNING_RATE = 0.01
NUM_EPOCHS = 40

# Generate data for the square function
sine_train_X, sine_train_F, sine_test_X, sine_test_F = generate_data(2 * np.pi, STEP_LENGTH, square_func, NOISE)

# Set up for varying the number of RBF units
max_rbf_units = 20
constant_variance = 0.1
sine_pred_F_units = []

for num_rbf_units in range(5, max_rbf_units + 15):
    #Handpicked initialisation
    mu_list = np.linspace(0, 2 * np.pi, num_rbf_units)
    #Random initialisation
    #mu_list = np.random.uniform(low=0, high=2*np.pi,size=(num_rbf_units, 1))
    variance_list = [constant_variance] * num_rbf_units
    rbf_network = RBF(mu_list, variance_list)
    rbf_network.seq_delta_training(sine_train_X, sine_train_F, LEARNING_RATE, NUM_EPOCHS)
    sine_pred_F_units.append(rbf_network.forward(sine_test_X))

# Set up for varying the variance
rbf_units = 15
sine_pred_F_var = []

for i in range(30):
    #Handpicked initialisation
    mu_list = np.linspace(0, 2 * np.pi, rbf_units)
    #Random initialisation
    #mu_list = np.random.uniform(low=0, high=2*np.pi,size=(rbf_units, 1))
    variance_list = [0.02 * (i + 1)] * rbf_units
    rbf_network = RBF(mu_list, variance_list)
    rbf_network.seq_delta_training(sine_train_X, sine_train_F, LEARNING_RATE, NUM_EPOCHS)
    sine_pred_F_var.append(rbf_network.forward(sine_test_X))

# Plotting the animations
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))  # Smaller figure size
print("units ", len(sine_pred_F_units), "var ", len(sine_pred_F_var))   
# Create the animation
ani = FuncAnimation(fig, update_plot, frames=min(len(sine_pred_F_units), len(sine_pred_F_var)),
                    fargs=(ax1, ax2, sine_test_X, sine_test_F, sine_pred_F_units, sine_pred_F_var),
                    interval=500)

plt.tight_layout()
ani.save('../out/task_1-2/rbf_varying_units_and_variance_seq_square_no_noise.gif', writer='pillow')
plt.show()
