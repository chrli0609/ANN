from rbf import RBF
from gen_data_func import *


import matplotlib.pyplot as plt
import numpy as np











########################## Sine ##########################



### SET PARAMETERS ###
#mu_list = np.linspace(0, 2*np.pi, 25)
mu_list = np.random.uniform(low=0, high=2*np.pi,size=(25, 1))
variance_list = [0.1] *len(mu_list)
NOISE = False
STEP_LENGTH = 0.1
LEARNING_RATE = 0.2
NUM_EPOCHS = 100

######################



#Generate Data for sine wave
sine_train_X, sine_train_F, sine_test_X, sine_test_F = generate_data(2*np.pi, STEP_LENGTH, sine_func, NOISE)


#sine_train_X = np.array([0.7, 0.8, 0.85, 1, 1.1, 1.2, 1.8, 1.9, 4.2, 4.6,5.1,5.2,5.3,5.4,5.5])
sine_train_X = np.concatenate((np.linspace(0, np.pi/2, 20), np.linspace(3*np.pi/2, np.pi*2, 20)))
#sine_train_X = np.linspace(0, np.pi/2, 20)
#sine_train_X = np.array([0.5])
rbf_network = RBF(mu_list, variance_list)
#rbf_network.plot_2d_weight_space((-10, 10), 100)

#Trains parameters of the RBF units
rbf_network.competitive_learning(sine_train_X, LEARNING_RATE, NUM_EPOCHS)
#Trains weights between units and output layer
rbf_network.seq_delta_training(sine_train_X, sine_train_F, LEARNING_RATE, NUM_EPOCHS)
#rbf_network.plot_1d_weight_space((-2, 2), 100)


sine_pred_F = rbf_network.forward(sine_test_X)


#rbf_network.plot_rbf_1d_inputs((0, 7), 100, sine_test_X, sine_pred_F, sine_test_F, "Sine")





'''
########################## Square ##########################
### SET PARAMETERS ###
#mu_list = np.arange(0, 2*np.pi, 0.2)
mu_list = [np.pi/4, 
            np.pi/2-0.5, np.pi/2-0.4, np.pi/2-0.3, np.pi/2-0.2, np.pi/2-0.1, np.pi/2, np.pi/2+0.1, np.pi/2+0.2, np.pi/2+0.3, np.pi/2+0.4, np.pi/2+0.5,
            3*np.pi/4,
            np.pi-0.5, np.pi-0.4, np.pi-0.3, np.pi-0.2, np.pi-0.1, np.pi, np.pi+0.1, np.pi+0.2, np.pi+0.3, np.pi+0.4, np.pi+0.5,
            5*np.pi/4,
            3*np.pi/2-0.5, 3*np.pi/2-0.4, 3*np.pi/2-0.3, 3*np.pi/2-0.2, 3*np.pi/2-0.1, 3*np.pi/2, 3*np.pi/2+0.1, 3*np.pi/2+0.2, 3*np.pi/2+0.3, 3*np.pi/2+0.4, 3*np.pi/2+0.5,
            7*np.pi/4,
            2*np.pi-0.3, 2*np.pi-0.2, 2*np.pi-0.1, 2*np.pi
        ]

#variance_list = [0.01] *len(mu_list)

variance_list = [0.1,
                0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
                0.1,
                0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
                0.1,
                0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
                0.1,
                0.001, 0.001, 0.001, 0.001
                ]



STEP_LENGTH = 0.1



#Generate Data for square wave
sine_train_X, sine_train_F, sine_test_X, sine_test_F = generate_data(2*np.pi, STEP_LENGTH, square_func, NOISE)

rbf_network = RBF(mu_list, variance_list)
#rbf_network.plot_2d_weight_space((-10, 10), 100)

rbf_network.competitive_learning(sine_train_X, LEARNING_RATE, NUM_EPOCHS)
rbf_network.batch_supervised_training(sine_train_X, sine_train_F)

sine_pred_F = rbf_network.forward(sine_test_X)

rbf_network.plot_rbf_1d_inputs((0, 7), 100, sine_test_X, sine_pred_F, sine_test_F, "Square")



'''