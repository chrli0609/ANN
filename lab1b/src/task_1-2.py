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






#three_d_plot(train_error_hidden_nodes_list)
animate_train_valid_error(train_error_hidden_nodes_list, valid_error_hidden_nodes_list)















