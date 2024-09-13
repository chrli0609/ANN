import numpy as np
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy



from model import MLP
from functions import *

 ## Format data
NUM_SAMPLES_PER_CLASS = 134
IN_DIM = 2

mA = np.array([1.0, 0.3])
sigmaA = 0.2
mB = np.array([0.0, -0.1])
sigmaB = 0.3


#subsampling_25_from_each_class
#subsampling_50_from_classA
#subsampling_point_2_lt_0_and_point_8_gt_0_from_A
init_W, X, T, X_test, T_test, color_list, color_list_test = generate_random_non_linear_input_and_weights(IN_DIM, NUM_SAMPLES_PER_CLASS, mA, mB, sigmaA, sigmaB, subsampling_25_from_each_class)



#Input data size
IN_DIM, num_input_samples = X.shape
LEARNING_RATE = 0.01
NUM_EPOCHS = 20
MAX_HIDDEN_NODES_TO_TRY = 55

#plot_data(X, color_list, X_test, color_list_test)



#define the model
#model = MLP(2, 4, 1, LEARNING_RATE)

#model.training(X, T, NUM_EPOCHS)







###########Investigate how the number of hidden nodes affect the performance###########

#Generate list of number of hidden nodes
hidden_nodes_list = np.arange(1, MAX_HIDDEN_NODES_TO_TRY, 1, dtype=int)


performance_mse_list = []
performance_acc_list = []
for hidden_nodes in hidden_nodes_list:
    print("Try", hidden_nodes, "number of hidden nodes")

    model = MLP(IN_DIM, hidden_nodes, 1, LEARNING_RATE)
    model.training(X, T, NUM_EPOCHS)

    #Test on validation data
    final_O, final_H = model.forward_pass(X_test)
    final_O = final_O.reshape((-1,))


    prediction = torch.from_numpy(final_O)
    target = torch.from_numpy(T_test)

    print("pred", prediction)
    print("target", target)

    loss_mse = nn.MSELoss()

    performance_mse = loss_mse(prediction, target)
    performance_acc = accuracy_score(prediction, target)


    performance_mse_list.append(performance_mse)
    performance_acc_list.append(performance_acc)

#performance_list.tolist()

print("performance_mse_list", performance_mse_list)
print("performance_acc_list", performance_acc_list)


fig, ax = plt.subplots(nrows=2, ncols=1)
fig.suptitle('Classification performance vs Number of neurons in hidden layer')


plt.subplot(2, 1, 1)
plt.plot(hidden_nodes_list, performance_mse_list, c='orange', label="MSELoss")
plt.xlabel("Number of Neurons in hidden layer")
plt.ylabel('Error')
plt.legend(loc="upper right")

plt.subplot(2, 1, 2)
plt.plot(hidden_nodes_list, performance_acc_list, c='blue', label="Accuracy Score")
plt.xlabel("Number of Neurons in hidden layer")
plt.ylabel('% accuracy')
plt.legend(loc="upper right")
plt.show()




'''
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Define the MLP model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Evaluate the model
loss, accuracy = model.evaluate(X, T, verbose=0)
print(f"Model Accuracy: {accuracy:.3f}")

# Test a prediction
sample = [3.6216,  8.6661,  -2.8073, -0.44699]
prediction = model.predict([sample])
print(f"Predicted Class: {round(prediction[0][0])}")

'''