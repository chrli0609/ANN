import numpy as np
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy


import math
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


from model import MLP
from functions import *

 ## Format data
NUM_SAMPLES_PER_CLASS = 134

mA = np.array([1.0, 0.3])
sigmaA = 0.2
mB = np.array([0.0, -0.1])
sigmaB = 0.3


#subsampling_25_from_each_class
#subsampling_50_from_classA
#subsampling_point_2_lt_0_and_point_8_gt_0_from_A
init_W, X, T, X_test, T_test, color_list = generate_random_non_linear_input_and_weights(IN_DIM, NUM_SAMPLES_PER_CLASS, mA, mB, sigmaA, sigmaB, subsampling_25_from_each_class)



#Input data size
IN_DIM, num_input_samples = X.shape
LEARNING_RATE = 0.01
NUM_EPOCHS = 20
MAX_HIDDEN_NODES_TO_TRY = 65

#plot_data(X, color_list)



#define the model
#model = MLP(2, 4, 1, LEARNING_RATE)

#model.training(X, T, NUM_EPOCHS)







###########Investigate how the number of hidden nodes affect the performance###########

#Generate list of number of hidden nodes
hidden_nodes_list = np.arange(1, MAX_HIDDEN_NODES_TO_TRY, 1, dtype=int)


train_error_hidden_nodes_list = []
valid_error_hidden_nodes_list = []
for hidden_nodes in hidden_nodes_list:
    print("Try", hidden_nodes, "number of hidden nodes")

    model = MLP(IN_DIM, hidden_nodes, 1, LEARNING_RATE)
    #model.training(X, T, NUM_EPOCHS)
    train_error, valid_error = model.training_w_valid(X, T, X_test, T_test, NUM_EPOCHS)

    train_error_hidden_nodes_list.append(train_error)
    valid_error_hidden_nodes_list.append(valid_error)


print("train_error_hidden_nodes_list", train_error_hidden_nodes_list)





















from mpl_toolkits.mplot3d.art3d import PolyCollection

def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]

ax = plt.figure().add_subplot(projection='3d')

# x corresponds to 64 points along the x-axis
x = np.linspace(0., 10., 64)

# Assume my_vec is of shape (64, 20), with 64 x-points and 20 lambda values
my_vec = np.random.rand(64, 20)  # You can replace this with your actual data

# verts[i] is a list of (x, y) pairs for each lambda value.
verts = [polygon_under_graph(x, my_vec[:, i]) for i in range(my_vec.shape[1])]

# Using 20 lambda values (lambdas = range(1, 21)) since my_vec has 20 columns
lambdas = range(1, 21)

# Define facecolors
facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(verts)))

# Create the PolyCollection object
poly = PolyCollection(verts, facecolors=facecolors, alpha=.7)
ax.add_collection3d(poly, zs=lambdas, zdir='y')

# Set limits and labels
ax.set(xlim=(0, 10), ylim=(1, 21), zlim=(0, 1),  # Adjust zlim based on data range
       xlabel='x', ylabel=r'$\lambda$', zlabel='probability')

plt.show()