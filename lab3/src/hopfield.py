import numpy as np
import random
import copy
from functions import *


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
        self.yippie = 0
        
        #Generate weight matrix
        self.weights = np.random.random((num_neurons, num_neurons))
        #If no self connection --> set diagonal elements to zero
        if not has_self_connections:
            np.fill_diagonal(self.weights, 0)


        #Generate list of Neurons
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(random.choice([-1, 1]))
        
        

    def recall(self, pattern, is_synch, max_iterations):
        self.yippie += 1
        #Initialize our pattern to the starting states of our neurons
        for i in range(self.num_neurons):
            self.neurons[i] = pattern[i]
        
        #update model for multiple time steps
        for it in range(max_iterations):

            if is_synch:
                if self.synch_update(it):
                    break

            else:
                #Make asynchronous update
                if self.asynch_update(it):
                    break

                #img = convert_1024_to_img_dim(self.neurons, img_dim=32)
                #visualize_img(image=img, image_num=iter, pattern_num=self.yippie, save_to_file=True)
                
                
    
    
    def asynch_update(self, iter_num):
        
        probe = np.random.permutation(range(self.num_neurons))


    
        counter = 0
        
        #Save the current states for comparison next iteration
        prev_states = copy.deepcopy(self.neurons)

        #For each neuron
        #for idx in probe:
        for idx in range(self.num_neurons):
            #print("counter",counter)
            #Checks if we have converged
            #if np.array_equal(self.neurons, prev_states):
            #    print("Optimal solution found for p", self.yippie, "at iteration", iter, "at step", counter)
            #    return True



            #Sum the contribution (weighted sums) from each neighbor
            weighted_sum = 0
            for j in range(len(self.weights[idx])):
                weighted_sum += self.weights[idx][j] * self.neurons[j]
            
    
            #Update the state of that neuron based on the majority vote of its neighbours
            self.neurons[idx] = sign(weighted_sum)



            #Visualize every hundred updates
            #if (counter%100 == 0):
            #    img = convert_1024_to_img_dim(self.neurons, img_dim=32)
            #    visualize_img(image=img, image_num=counter+iter_num*10000, pattern_num=self.yippie, save_to_file=True)

            counter += 1


        #img = convert_1024_to_img_dim(self.neurons, img_dim=32)
        #visualize_img(image=img, image_num=iter_num, pattern_num=self.yippie, save_to_file=True)


        #print("==============================", iter_num, "==================================")
        #print("prev_states\t\t", np.array(prev_states), "energy\t", self.compute_energy(prev_states))
        #print("self.neurons\t\t", np.array(self.neurons), "energy\t", self.compute_energy(self.neurons))
        #print("elementwise mult\t", np.multiply(np.array(self.neurons), np.array(prev_states)))

        if np.array_equal(self.neurons, prev_states):
            return True
        else:
            return False
        


    

    def synch_update(self, iter_num):


        new_neuron_states = vsign(np.matmul(self.weights, self.neurons))

        
        """img = convert_1024_to_img_dim(new_neuron_states, img_dim=32)
        visualize_img(image=img, image_num=iter_num, pattern_num=self.yippie, save_to_file=True)"""



        print("==============================", iter_num, "==================================")
        print("prev_states\t\t", np.array(self.neurons), "energy\t", self.compute_energy(self.neurons))
        print("self.neurons\t\t", np.array(new_neuron_states), "energy\t", self.compute_energy(new_neuron_states))
        print("elementwise mult\t", np.sum(np.multiply(np.array(new_neuron_states), np.array(self.neurons))))



        if np.array_equal(self.neurons, new_neuron_states):
            #print("Optimal solution found for p", self.yippie)
            return True
        else:
            self.neurons = copy.deepcopy(new_neuron_states)
            return False


                
    
    def train(self, X):

        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):

                inner_prod = 0
                for mu in range(len(X)):
                    inner_prod += X[mu][i] * X[mu][j]

                self.weights[i][j] = inner_prod / self.num_neurons

        
        if not self.has_self_connections:
             np.fill_diagonal(self.weights, 0)

    
    def compute_energy(self, list_of_neuron_states):
        
        E = 0
        for i in range(len(list_of_neuron_states)):
            for j in range(len(list_of_neuron_states)):
                E -= self.weights[i][j] * list_of_neuron_states[i]*list_of_neuron_states[j]

        return E
    