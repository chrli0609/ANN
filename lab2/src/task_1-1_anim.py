from rbf import RBF
import numpy as np

def sine_func(x):
    return np.sin(2 * x)

def generate_data(x_end, step_length, function):
    train_X = (np.arange(0, x_end, step_length)).reshape(-1, 1)
    test_X = (np.arange(0.05, x_end, step_length)).reshape(-1, 1)

    train_F = np.array([function(x) for x in train_X])
    test_F = np.array([function(x) for x in test_X])

    return train_X, train_F, test_X, test_F

mu_list = np.arange(0.1, 1.6, 0.1)  # Mu values from 0.1 to 1.5
variance_list = [0.1] * len(mu_list)
STEP_LENGTH = 0.1

# Generate data for the sine wave
sine_train_X, sine_train_F, sine_test_X, sine_test_F = generate_data(2 * np.pi, STEP_LENGTH, sine_func)

# Initialize RBF network
rbf_network = RBF(mu_list, variance_list)

# Train RBF network
rbf_network.batch_supervised_training(sine_train_X, sine_train_F)

# Animate RBF network with subplots
rbf_network.plot_rbf_1d_inputs_animated((0, 7), 100, sine_test_F, sine_test_F, "Sine", sine_train_X, sine_train_F)




