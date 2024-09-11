import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



def activation_func(x_mat):
    return 2 / (1 + np.power(math.e, x_mat)) - 1


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

            print("O", O.shape)

            delta_o, delta_h = self.backward_pass(X, O, H, T)
            print("delta_o", delta_o.shape)
            print("delta_h", delta_h.shape)

            self.weight_update(X, H, delta_o, delta_h)

            print("W:\n", self.W)
            print("V:\n", self.V)








    def forward_pass(self, X):

        _, num_input_samples = X.shape
        

        print("W.T", self.W.T.shape)
        print("X", X.shape)
        print("V.T", self.V.T.shape)
        
        
        #Append X with one row of ones for bias
        X_bias = np.concatenate((X, np.ones((1, num_input_samples))), axis=0)

        H = activation_func(np.matmul(self.W.T, X_bias))

        #Append H with one row of ones for bias
        H_bias = np.concatenate((H, np.ones((1, num_input_samples))), axis=0)

        print("H", H.shape)

        


        print("aeoringoperpaghaoo;apegboarhoaego;hergaeh;ogh;oaer")

        #softmax_func = nn.Softmax(dim=0) 
        no_softmax = np.matmul(self.V.T, H_bias)
        print("no_softmax", no_softmax)
        
        O = F.softmax(no_softmax, dim=0)
        print("O", O)
       

        #O = activation_func(np.matmul(self.V.T, H_bias))
        

        return O, H



    def backward_pass(self, X, out, hout, targets):

        _, num_input_samples = X.shape


        #O = out
        #H = hout
        hout_bias = np.concatenate((hout, np.ones((1, num_input_samples))), axis=0)


        #softmax_func = nn.Softmax(dim=0)
        #delta_o = np.multiply((out-targets), softmax_func(out))
        delta_o = np.multiply((out-targets), F.softmax(out, dim=0))

        print("V", self.V.shape)
        print("delta_o", delta_o.shape)
        

        delta_h = np.multiply(np.matmul(self.V, delta_o), d_activation_func(hout_bias))
        
        #Remove extra row that was added to handle bias
        delta_h = delta_h[:-1, :]



        return delta_o, delta_h

    def weight_update(self, X, H, delta_o, delta_h):
        _, num_input_samples = X.shape

        print("delta_h", delta_h.shape)
        print("X", X.shape)
        print("self.eta", self.eta)
        print("delta_o", delta_o.shape)
        print("H", H.shape)
        #Fill
        X_bias = np.concatenate((X, np.ones((1, num_input_samples))), axis=0)
        H_bias = np.concatenate((H, np.ones((1, num_input_samples))), axis=0)

        print("X_bias", X_bias.shape)
        print("H_bias", H_bias.shape)
        print("self.V", self.V.shape)

        self.W += -self.eta * np.matmul(delta_h, X_bias.T).T
        self.V += -self.eta * np.matmul(delta_o, H_bias.T).T




def softmax(x):
    """Compute the softmax of vector x."""
    exps = np.exp(x)
    return exps / np.sum(exps)