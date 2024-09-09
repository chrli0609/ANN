import numpy as np
import torch
import torch.nn as nn


from model import MLP
from functions import *

 ## Format data
mA = np.array([1.0, 0.3])
sigmaA = 0.2
mB = np.array([0.0, -0.1])
sigmaB = 0.3


#subsampling_25_from_each_class
#subsampling_50_from_classA
#subsampling_point_2_lt_0_and_point_8_gt_0_from_A
init_W, X, T, X_test, T_test, color_list = generate_random_non_linear_input_and_weights(IN_DIM, NUM_SAMPLES_PER_CLASS, mA, mB, sigmaA, sigmaB, subsampling_point_2_lt_0_and_point_8_gt_0_from_A)



#Input data size
IN_DIM, num_input_samples = X.shape
LEARNING_RATE = 0.0001
NUM_EPOCHS = 40

#plot_data(X, color_list)



#define the model
model = MLP(2, 4, 1, LEARNING_RATE)

model.training(X, T, NUM_EPOCHS)







###########Investigate how the number of hidden nodes affect the performance###########

#Generate list of number of hidden nodes
hidden_nodes_list = np.arange(1, 20, 1, dtype=int)


performance_list = []
for hidden_nodes in hidden_nodes_list:
    print("Try", hidden_nodes, "number of hidden nodes")

    model = MLP(IN_DIM, hidden_nodes, 1, LEARNING_RATE)
    model.training(X, T, NUM_EPOCHS)

    #Test on validation data
    final_O, final_H = model.forward_pass(X_test)


    prediction = torch.from_numpy(final_O)
    target = torch.from_numpy(T_test)
    

    loss = nn.MSELoss()
    performance = loss(prediction, target)


    performance_list.append(performance)

#performance_list.tolist()

print("performance_list", performance_list)

plt.plot(hidden_nodes_list, performance_list)
plt.show()




