import numpy as np
import random
import copy

#from functions import *


def sign(x):

    if x >= 0:
        return 1
    else:
        return -1


def vsign(x_vec):
    ret_vec = []
    for i in range(len(x_vec)):
        if x_vec[i] >= 0:
            ret_vec.append(1)
        else:
            ret_vec.append(-1)


    return ret_vec
    



class Hopfield:
    def __init__(self, num_neurons, has_self_connections):
        self.num_neurons = num_neurons
        self.has_self_connections = has_self_connections
        
        #Generate weight matrix
        self.weights = np.random.random((num_neurons, num_neurons))
        #If no self connection --> set diagonal elements to zero
        if not has_self_connections:
            np.fill_diagonal(self.weights, 0)


        #Generate list of Neuron objects
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(random.choice([-1, 1]))


    def recall(self, pattern, is_synch, max_iterations):
        
        #Initialize our pattern to the starting states of our neurons
        for i in range(self.num_neurons):
            self.neurons[i] = pattern[i]


        prev_states = np.zeros_like(self.neurons)
        #update model for multiple time steps
        for it in range(max_iterations):


            #make comparison here
            if np.array_equal(self.neurons, prev_states):
                break


            #Save the current states for comparison next iteration
            prev_states = copy.deepcopy(self.neurons)

            if is_synch:
                self.synch_update()

            else:
                #Make asynchronous update
                self.asynch_update()
                
                
    
    
    def asynch_update(self):
        
        probe = np.random.permutation(len(self.num_neurons))


        #For each neuron
        #for i in range(len(self.weights)):
        for idx in probe:

            #Sum the contribution (weighted sums) from each neighbor
            weighted_sum = 0
            for j in range(len(self.weights[idx])):
                weighted_sum += self.weights[idx][j] * self.neurons[idx].state 
        
            
            #Update the state of that neuron based on the majority vote of its neighbours
            self.neurons[idx] = sign(weighted_sum)

    

    def synch_update(self):
        #Matrix mult ftw
        self.neurons = vsign(np.matmul(self.weights, self.neurons))

                

    
    def train(self, X):

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):

                inner_prod = 0
                for mu in range(len(X)):
                    inner_prod += X[mu][i] * X[mu][j]

                self.weights[i][j] = inner_prod / self.num_neurons

        
        if not self.has_self_connections:
             np.fill_diagonal(self.weights, 0)

