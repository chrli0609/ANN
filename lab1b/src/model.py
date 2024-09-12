import math
import numpy as np
import torch
import torch.nn as nn

def activation_func(x_mat):
    return 2 / (1 + np.power(math.e, -x_mat)) - 1


def d_activation_func(activation_func_x_mat):
    return np.multiply((1 + activation_func_x_mat), (1 - activation_func_x_mat)) / 2



class MLP():
    def __init__(self, IN_DIM, NUM_IN_NODES, OUT_DIM, LEARNING_RATE):
        self.in_dim = IN_DIM
        self.out_dim = OUT_DIM

        #Add extra row at bottom for bias
        self.W = np.random.rand(IN_DIM+1, NUM_IN_NODES)
        self.V = np.random.rand(NUM_IN_NODES+1, OUT_DIM)

        self.eta = LEARNING_RATE


    def training(self, X, T, NUM_EPOCHS):
        for epoch in range(NUM_EPOCHS):
            O, H = self.forward_pass(X)


            delta_o, delta_h = self.backward_pass(X, O, H, T)

            self.weight_update(X, H, delta_o, delta_h)



    def training_w_valid(self, X, T, X_test, T_test, NUM_EPOCHS):
        valid_error_list = []
        train_error_list = []
        for epoch in range(NUM_EPOCHS):
            O, H = self.forward_pass(X)

            delta_o, delta_h = self.backward_pass(X, O, H, T)

            self.weight_update(X, H, delta_o, delta_h)



            #Define loss
            mse_loss = nn.MSELoss()

            #Get training error
            O_tensor = torch.tensor(O)
            T_tensor = torch.tensor(T)
            train_error_list.append(mse_loss(O_tensor, T_tensor).tolist())


            #Get validation error
            O_valid, _ = self.forward_pass(X_test)

            O_valid_tensor = torch.tensor(O_valid)
            T_test_tensor = torch.tensor(T_test)
            valid_error_list.append(mse_loss(O_valid_tensor, T_test_tensor).tolist())

        

        return train_error_list, valid_error_list




    def compute_mse(self, X, T):


        return mse_loss(X, T)




    def forward_pass(self, X):

        _, num_input_samples = X.shape
        
        
        
        #Append X with one row of ones for bias
        X_bias = np.concatenate((X, np.ones((1, num_input_samples))), axis=0)

        H = activation_func(np.matmul(self.W.T, X_bias))

        #Append H with one row of ones for bias
        H_bias = np.concatenate((H, np.ones((1, num_input_samples))), axis=0)

        O = activation_func(np.matmul(self.V.T, H_bias))
        

        return O, H



    def backward_pass(self, X, out, hout, targets):

        _, num_input_samples = X.shape


        #O = out
        #H = hout
        hout_bias = np.concatenate((hout, np.ones((1, num_input_samples))), axis=0)

        delta_o = np.multiply((out-targets), d_activation_func(out))


        delta_h = np.multiply(np.matmul(self.V, delta_o), d_activation_func(hout_bias))
        
        #Remove extra row that was added to handle bias
        delta_h = delta_h[:-1, :]



        return delta_o, delta_h

    def weight_update(self, X, H, delta_o, delta_h):
        _, num_input_samples = X.shape

 
        #Fill
        X_bias = np.concatenate((X, np.ones((1, num_input_samples))), axis=0)
        H_bias = np.concatenate((H, np.ones((1, num_input_samples))), axis=0)


        self.W += -self.eta * np.matmul(delta_h, X_bias.T).T
        self.V += -self.eta * np.matmul(delta_o, H_bias.T).T




