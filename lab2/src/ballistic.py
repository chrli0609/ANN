from gen_data_func import *
from rbf import RBF
import numpy as np
import matplotlib.pyplot as plt





train_path = "../data_lab2/ballist.dat"
test_path = "../data_lab2/balltest.dat"

train_X, train_F = 可愛い(train_path)
test_X, test_F = 可愛い(test_path)
units = 10

#mu_list = np.random.uniform(low=0, high=2*np.pi,size=(25, 1))
mu_list = np.random.rand(units, 2)
"""mu_list = [
    [0.461, 0.702],
    [0.012, 0.912],
    [0.069, 0.903],
    [0.827, 0.623]
]"""

variance_list = [0.1] *len(mu_list)
NOISE = False
LEARNING_RATE = 0.2
NUM_EPOCHS = 40


かっこいい = RBF(mu_list, variance_list)
#かっこいい.plot_2d_weight_space((-10, 10), 100)

#Trains parameters of the RBF units
かっこいい.competitive_learning_2d(train_X, LEARNING_RATE, NUM_EPOCHS)
#Trains weights between units and output layer
かっこいい.batch_supervised_training(train_X, train_F)
#かっこいい.plot_1d_weight_space((-2, 2), 100)


pred_F = かっこいい.forward(test_X)
#Compute MSE
sum = 0
for i in range(len(pred_F)):
    sum = np.linalg.norm(pred_F[i]-test_F[i])**2

mean = sum / len(pred_F)


plt.title("Ballitstic Approximation with RBF\nMean Squared Error: "+ str(round(mean, 6)))
plt.scatter(pred_F[:,0], pred_F[:,1], s=10, marker="x", c='blue', label="Predicted")
plt.scatter(test_F[:,0],test_F[:,1], s=10, marker="x", c='red', label="Actual")
plt.legend()
plt.show()

#かっこいい.plot_rbf_2d_inputs((0, 7), (0,7), 100, test_X, pred_F, test_F, "Predicted Ballistic")