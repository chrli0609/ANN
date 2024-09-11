import numpy as np
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy



from model_softmax import MLP
from functions import *

 ## Format data
mA = np.array([1.0, 0.3])
sigmaA = 0.2
mB = np.array([0.0, -0.1])
sigmaB = 0.3


#subsampling_25_from_each_class
#subsampling_50_from_classA
#subsampling_point_2_lt_0_and_point_8_gt_0_from_A
init_W, X, T, X_test, T_test, color_list = generate_random_non_linear_input_and_weights(IN_DIM, NUM_SAMPLES_PER_CLASS, mA, mB, sigmaA, sigmaB, subsampling_25_from_each_class)


T = single_to_double_T(T)
T_test = single_to_double_T(T_test)


#Input data size
IN_DIM, num_input_samples = X.shape
LEARNING_RATE = 0.0001
NUM_EPOCHS = 40

#plot_data(X, color_list)





###########Investigate how the number of hidden nodes affect the performance###########

#Generate list of number of hidden nodes
hidden_nodes_list = np.arange(1, 20, 1, dtype=int)


performance_mse_list = []
performance_acc_list = []
for hidden_nodes in hidden_nodes_list:
    print("Try", hidden_nodes, "number of hidden nodes")

    model = MLP(IN_DIM, hidden_nodes, 2, LEARNING_RATE)
    model.training(X, T, NUM_EPOCHS)

    #Test on validation data
    final_O, final_H = model.forward_pass(X_test)
    print("final_O", final_O)
    #final_O = final_O.reshape((-1,))

    print("sum row ", np.sum(final_O, axis=1))

    


    prediction = torch.from_numpy(final_O)
    target = torch.from_numpy(T_test)
    
    print("pred", prediction)
    print("target", target)



    loss_mse = nn.MSELoss()
    loss_acc = BinaryAccuracy()

    performance_mse = loss_mse(prediction, target)
    performance_acc = loss_acc(prediction, target)


    performance_mse_list.append(performance_mse)
    performance_acc_list.append(performance_acc)

#performance_list.tolist()

print("performance_mse_list", performance_mse_list)
print("performance_acc_list", performance_acc_list)


plt.plot(hidden_nodes_list, performance_mse_list, label="MSELoss")
plt.plot(hidden_nodes_list, performance_acc_list, label="Accuracy Score")
plt.legend(loc="upper right")
plt.show()




