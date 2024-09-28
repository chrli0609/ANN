import numpy as np
import random
import copy

#from functions import *


def sign(x):
    #ret_vec = copy.deepcopy(x)

    #ret_vec[ret_vec >= 0] = 1
    #ret_vec[ret_vec < 0] = -1
    if x >= 0:
        return 1
    else:
        return -1


'''
    ret_vec = []
    for i in range(len(x)):
        if x[i] >= 0:
            ret_vec.append(1)
        else:
            ret_vec.append(-1)
'''
    


class Neuron:
    def __init__(self, state):

        if abs(state) != 1:
            raise("Invalid state for neuron", state)
        
        self.state = state
        



class Hopfield:
    def __init__(self, num_neurons, self_connection):
        self.num_neurons = num_neurons
        self.self_connection = self_connection
        
        #Generate weight matrix
        self.weights = np.random.random((num_neurons, num_neurons))
        #If no self connection --> set diagonal elements to zero
        if not self_connection:
            np.fill_diagonal(self.weights, 0)


        #Generate list of Neuron objects
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(random.choice([-1, 1])))


    def recall(self, pattern):

        x = np.zeros_like(pattern)
        for i in range(len(x)):

            #Weighted sum
            weighted_sum = 0
            for j in range(len(self.weights[i])):

                if self.self_connection and i == j:
                    continue

                weighted_sum += self.weights[i][j] * pattern[j]
            
            #get signed bipolar value
            x[i] = sign(weighted_sum)
        
        return x
    
    def synchronous_training(self, X):

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):

                inner_prod = 0
                for mu in range(len(X)):
                    inner_prod += X[mu][i] * X[mu][j]

                self.weights[i][j] = inner_prod / self.num_neurons


    def asynchronous_training(self):
        return