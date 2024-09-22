import numpy as np
import math
from rbf import RBF
import matplotlib.pyplot as plt





def sine_func(x):
    return np.sin(2*x)

def square_func(x):
    return 1 if sine_func(x) > 0 else -1


def generate_data(x_end, step_length, function, noise):

    train_X = ((np.arange(0 , x_end, step_length)).T).reshape(-1,1)
    test_X = ((np.arange(0.05, x_end, step_length)).T).reshape(-1,1)
    

    N_train_samples = len(train_X)
    N_test_samples = len(test_X)



    train_F = np.zeros((N_train_samples, 1))
    test_F = np.zeros((N_test_samples, 1))

    for i in range(len(train_X)):
        train_F[i] = function(train_X[i])

    for i in range(len(test_X)):
        test_F[i] = function(test_X[i])


    if (noise):
        train_F += np.random.normal(0, 0.1, size=(N_train_samples, 1))
        test_F += np.random.normal(0, 0.1, size=(N_test_samples,1))

    print("train_X", train_X.shape)

    return train_X, train_F, test_X, test_F


########################## Sine ##########################



### SET PARAMETERS ###
mu_list = np.arange(0, 2*np.pi, 0.22)
variance_list = [0.1] *len(mu_list)

STEP_LENGTH = 0.1
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
NOISE = True

############################################


'''
#Generate Data for sine wave
sine_train_X, sine_train_F, sine_test_X, sine_test_F = generate_data(2*np.pi, STEP_LENGTH, sine_func)


rbf_network = RBF(mu_list, variance_list)
#rbf_network.plot_2d_weight_space((-10, 10), 100)

rbf_network.seq_delta_training(sine_train_X, sine_train_F, LEARNING_RATE, NUM_EPOCHS)
#rbf_network.plot_1d_weight_space((-2, 2), 100)


sine_pred_F = rbf_network.forward(sine_test_X)


rbf_network.plot_rbf_1d_inputs((0, 7), 100, sine_test_X, sine_pred_F, sine_test_F, "Noisy Sine")

'''

############################################


########################## Square ##########################
### SET PARAMETERS ###
mu_list = np.arange(0, 2*np.pi, 1)
'''
mu_list = [np.pi/4, 
            np.pi/2-0.5, np.pi/2-0.4, np.pi/2-0.3, np.pi/2-0.2, np.pi/2-0.1, np.pi/2, np.pi/2+0.1, np.pi/2+0.2, np.pi/2+0.3, np.pi/2+0.4, np.pi/2+0.5,
            3*np.pi/4,
            np.pi-0.5, np.pi-0.4, np.pi-0.3, np.pi-0.2, np.pi-0.1, np.pi, np.pi+0.1, np.pi+0.2, np.pi+0.3, np.pi+0.4, np.pi+0.5,
            5*np.pi/4,
            3*np.pi/2-0.5, 3*np.pi/2-0.4, 3*np.pi/2-0.3, 3*np.pi/2-0.2, 3*np.pi/2-0.1, 3*np.pi/2, 3*np.pi/2+0.1, 3*np.pi/2+0.2, 3*np.pi/2+0.3, 3*np.pi/2+0.4, 3*np.pi/2+0.5,
            7*np.pi/4,
            2*np.pi-0.3, 2*np.pi-0.2, 2*np.pi-0.1, 2*np.pi
        ]
'''
variance_list = [1] *len(mu_list)
'''
variance_list = [0.1,
                0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
                0.1,
                0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
                0.1,
                0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
                0.1,
                0.001, 0.001, 0.001, 0.001
                ]

'''

STEP_LENGTH = 0.1
NOISE = True
LEARNING_RATE = 0.001
NUM_EPOCHS = 20


#Generate Data for square wave
sine_train_X, sine_train_F, sine_test_X, sine_test_F = generate_data(2*np.pi, STEP_LENGTH, square_func, NOISE)

rbf_network = RBF(mu_list, variance_list)
#rbf_network.plot_2d_weight_space((-10, 10), 100)

rbf_network.seq_delta_training(sine_train_X, sine_train_F, LEARNING_RATE, NUM_EPOCHS)

sine_pred_F = rbf_network.forward(sine_test_X)

rbf_network.plot_rbf_1d_inputs((0, 7), 100, sine_test_X, sine_pred_F, sine_test_F, "Noisy Square")