import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


from model import MLP
from functions import *

# Data and parameters
NUM_SAMPLES_PER_CLASS = 100
mA = np.array([1.0, 0.3])
sigmaA = 0.2
mB = np.array([0.0, -0.1])
sigmaB = 0.3

# Generate random data
IN_DIM = 2
NUM_EPOCHS = 30
LEARNING_RATE = 0.01
MAX_HIDDEN_NODES_TO_TRY = 50

# Generate data and weights
init_W, X, T, X_test, T_test, color_list = generate_random_non_linear_input_and_weights(
    IN_DIM, NUM_SAMPLES_PER_CLASS, mA, mB, sigmaA, sigmaB, subsampling_25_from_each_class
)

print("X", X)
print("T", T)


print("X_text", X_test)
print("T_text", T_test)

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



print("train_error_hidden_nodes_list", train_error_hidden_nodes_list)
print("valid_error_hidden_nodes_list", valid_error_hidden_nodes_list)




# Call the animation function when needed
animate_training_validation_errors(NUM_EPOCHS, hidden_nodes_list, train_error_hidden_nodes_list, valid_error_hidden_nodes_list)

'''
list1 = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
]

list2 = [[2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
]
        
        
        



animate_training_validation_errors(20, [1, 2, 3, 4, 6], list1, list2)
'''