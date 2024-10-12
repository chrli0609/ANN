from util import *
import random
import numpy as np

class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28,28], is_top=False, n_labels=10, batch_size=10):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end. 
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """
       
        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom : self.image_size = image_size
        
        self.is_top = is_top

        if is_top : self.n_labels = 10

        self.batch_size = batch_size        
                
        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))

        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible,self.ndim_hidden))

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))
        
        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0        
        
        self.weight_v_to_h = None
        
        self.weight_h_to_v = None

        self.learning_rate = 0.01
        
        self.momentum = 0.7

        self.print_period = 5000
        #self.print_period = 60

        self.debug_weights = []
        
        self.rf = { # receptive-fields. Only applicable when visible layer is input data
            "period" : 5000, # iteration period to visualize
            #"period" : 60, # iteration period to visualize
            "grid" : [5,5], # size of the grid
            "ids" : np.random.randint(0,self.ndim_hidden,25) # pick some random hidden units
            }
        
        return

        
    def cd1(self, visible_trainset, n_iterations=10000):
        
        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print ("learning CD1")

        #visible_trainset = visible_trainset[0:100,:]
        
        #print("visible_trainset", visible_trainset.__class__)
        
        n_samples = visible_trainset.shape[0]


        if n_iterations * self.batch_size != n_samples:
            print("AN ERROR HAS OCCURREEEEEEDDDD: n_iterations * self.batch_size != n_samples")
        
        #v_states_0 = visible_trainset
        #h_states_0 = [None] * n_samples
        #v_states_1 = [None] * n_samples
        #h_states_1 = [None] * n_samples

        for it in range(n_iterations):
            
            #print("visible_trainset", visible_trainset.shape)
            v_batch_prob_0 = visible_trainset[it*self.batch_size:(it+1)*self.batch_size][:]

            v_batch_states_0 = sample_binary(v_batch_prob_0)

            
            
            #Awake
            
            h_batch_prob_0, h_batch_states_0 = self.get_h_given_v(v_batch_states_0)
            #for i in range(self.batch_size):
            #    h_states_0[i] = h_batch_states_0
            

            #Asleep
            v_batch_prob_1, v_batch_states_1 = self.get_v_given_h(h_batch_states_0)
            #for i in range(self.batch_size):
            #    v_states_1[i] = v_batch_states



            #Awake 2
            h_batch_prob_1, h_batch_states_1 = self.get_h_given_v(v_batch_states_1)
            #for i in range(self.batch_size):
            #    h_states_1[i] = h_batch_states_1


            
            #self.update_params(v_states_0, h_states_0, v_states_1, h_states_1)
            self.update_params(v_batch_prob_0, h_batch_states_0, v_batch_prob_1, h_batch_prob_1, n_samples)



	    # [DONE TASK 4.1] run k=1 alternating Gibbs sampling : v_0 -> h_0 ->  v_1 -> h_1.
            # you may need to use the inference functions 'get_h_given_v' and 'get_v_given_h'.
            # note that inference methods returns both probabilities and activations (samples from probablities) and you may have to decide when to use what.

            # [DONE TASK 4.1] update the parameters using function 'update_params'
            
            # visualize once in a while when visible layer is input images
            
            if it % self.rf["period"] == 0 and self.is_bottom:
                
                viz_rf(weights=self.weight_vh[:,self.rf["ids"]].reshape((self.image_size[0],self.image_size[1],-1)), it=it, grid=self.rf["grid"])

            # print progress
            
            if it % self.print_period == 0 :

                print ("iteration=%7d recon_loss=%4.4f"%(it, np.linalg.norm(visible_trainset - visible_trainset)))
        
        return
    

    def update_params(self,v_0,h_0,v_k,h_k, n_samples):

        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [DONE TASK 4.1] get the gradients from the arguments (replace the 0s below) and update the weight and bias parameters


        #print("Updating params")
        #print("v_0\n", v_0.tolist(), "\n", np.sum(v_0))
        #print("h_0\n", h_0, "\n", np.sum(h_0))
        #print("v_k\n", v_k, "\n", np.sum(v_k))
        #print("h_k\n", h_k, "\n", np.sum(h_k))

        """
        # Compute the average difference in biases across all samples (vectorized)
        delta_bias_v = np.mean(v_0 - v_k, axis=0)  # Visible bias update
        delta_bias_h = np.mean(h_0 - h_k, axis=0)  # Hidden bias update

        # Compute the positive and negative gradients for the entire batch using matrix multiplication
        # This effectively replaces the outer products for each sample
        positive_grad = np.dot(v_0.T, h_0)  # Data-dependent term
        negative_grad = np.dot(v_k.T, h_k)  # Model-dependent term

        # Compute the delta for weights, averaging the difference across all samples
        delta_weight = (positive_grad - negative_grad) / self.batch_size
        """

        delta_bias_v = np.zeros(self.ndim_visible)
        delta_bias_h = np.zeros(self.ndim_hidden)
        
        delta_weight = np.zeros((v_0.shape[1], h_0.shape[1]))

        # Update rule for the weights based on Contrastive Divergence
        # Calculate the outer product for each sample and average over all samples
        for n in range(self.batch_size):

            #Bias
            delta_bias_v += (v_0[n] - v_k[n]) 
            delta_bias_h += (h_0[n] - h_k[n])
                

            positive_grad = np.outer(v_0[n], h_0[n])  # Data-dependent term
            negative_grad = np.outer(v_k[n], h_k[n])  # Model-dependent term


            delta_weight += (positive_grad - negative_grad)
        
        #print("norm(delta_weights)", np.linalg.norm(delta_weight))

        

        
        self.delta_bias_v += delta_bias_v / n_samples
        self.delta_weight_vh += delta_weight / n_samples
        self.delta_bias_h += delta_bias_h  / n_samples
        
        self.bias_v += self.delta_bias_v
        self.weight_vh += self.delta_weight_vh
        self.bias_h += self.delta_bias_h

        #================================L2 NORM BETWEEN DELTA W AND ORIGIN EACH SAMPLE =================================
        self.debug_weights.append(np.linalg.norm(delta_weight))
        
        return

    #Takes input as binary states and produces probabilities and binary states
    def get_h_given_v(self,visible_minibatch):
        
        """Compute probabilities p(h|v) and activations h ~ p(h|v) 

        Uses undirected weight "weight_vh" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_vh is not None

        n_samples = visible_minibatch.shape[0]


        # [DONE TASK 4.1] compute probabilities and activations (samples from probabilities) of hidden layer (replace the zeros below) 

        # Matrix-based implementation for all samples
        # Compute the weighted sum for all hidden neurons for all samples using matrix multiplication
        # Each row corresponds to one sample, each column to one hidden neuron
        weighted_sum_all_samples = np.dot(visible_minibatch, self.weight_vh) + self.bias_h


        # Compute the probability of activation for all hidden neurons across all samples (vectorized)
        #probability_on_all_hidden_neurons_all_samples = 1 / (1 + np.exp(-weighted_sum_all_samples))
        probability_on_all_hidden_neurons_all_samples = sigmoid(weighted_sum_all_samples)
        #print("probability_on_all_hidden_neurons_all_samples", probability_on_all_hidden_neurons_all_samples)

        # Generate random values for each hidden neuron of every sample for activation decision
        #random_values = np.random.rand(n_samples, self.ndim_hidden)

        # Determine activation based on probability and random values (vectorized)
        #activation_all_hidden_neurons_all_samples = (random_values < probability_on_all_hidden_neurons_all_samples).astype(int)

        activation_all_hidden_neurons_all_samples = sample_binary(probability_on_all_hidden_neurons_all_samples)

        # Since we already have matrix-based results, no need for appending
        # Directly return the probability and activation matrices
        return probability_on_all_hidden_neurons_all_samples, activation_all_hidden_neurons_all_samples

    #Takes input as binary states and produces probabilities and binary states
    def get_v_given_h(self,hidden_minibatch):
        
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            

            # [DONE TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass below). \
            # Note that this section can also be postponed until TASK 4.2, since in this task, stand-alone RBMs do not contain labels in visible layer.
            
            #1. Compute updated weighted sums from lvl 1 neighbors

            #2. Separate pen and label units 

            #3. Apply probability + sign to pen units

            #4. Apply softmax to label units


            # Compute weighted sums for all samples and all visible neurons at once using matrix multiplication
            # Each row of the result corresponds to a sample
            weighted_sum_all_samples = np.dot(hidden_minibatch, self.weight_vh.T) + self.bias_v
            #print("weighted_sum_all_samples", weighted_sum_all_samples.shape)

            


            #Split pen and label units
            pen_units_weighted_sum = weighted_sum_all_samples[:, :-self.n_labels]
            label_units_weighted_sum = weighted_sum_all_samples[:, -self.n_labels:]


            ######################################
            ###### Handle Penultimate Units ######
            ######################################
            # Compute the probability of activation for all visible neurons for all samples (vectorized)

            #probability_on_all_pen_neurons_all_samples = 1 / (1 + np.exp(-pen_units_weighted_sum))
            probability_on_all_pen_neurons_all_samples = sigmoid(pen_units_weighted_sum)

            # Generate random values for each visible neuron of every sample for activation decision
            #random_values = np.random.rand(n_samples, self.ndim_visible-self.n_labels)
            # Determine activation based on probability and random values (vectorized)
            #activation_all_pen_units = (random_values < probability_on_all_pen_neurons_all_samples).astype(int)

            activation_all_pen_units = sample_binary(probability_on_all_pen_neurons_all_samples)


            ################################
            ###### Handle Label Units ######
            ################################
            
            probability_all_label_units_all_samples = softmax(label_units_weighted_sum)
                        
            # Step 1: Find the index of the maximum value along the second axis (axis=1)
            #max_indices = np.argmax(probability_all_label_units_all_samples, axis=1)

            # Step 2: Create a one-hot encoded matrix with the same shape as the original probabilities
            #activation_all_label_units = np.zeros_like(probability_all_label_units_all_samples)

            # Step 3: Set the max index to 1 for each sample
            #activation_all_label_units[np.arange(probability_all_label_units_all_samples.shape[0]), max_indices] = 1

            activation_all_label_units = sample_categorical(probability_all_label_units_all_samples)



            #####################################################
            ###### Concatenate Penultimate and Label units ######
            #####################################################
            probability_all_visible_neurons_all_samples = np.concatenate((probability_on_all_pen_neurons_all_samples, probability_all_label_units_all_samples), axis=1)
            activation_all_visible_neurons_all_samples = np.concatenate((activation_all_pen_units, activation_all_label_units), axis=1)



            # Return the probabilities and activations for all samples
            return probability_all_visible_neurons_all_samples, activation_all_visible_neurons_all_samples
            

            
        else:
                        
            # [DONE TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass and zeros below)             

            # Compute weighted sums for all samples and all visible neurons at once using matrix multiplication
            # Each row of the result corresponds to a sample
            weighted_sum_all_samples = np.dot(hidden_minibatch, self.weight_vh.T) + self.bias_v

            # Compute the probability of activation for all visible neurons for all samples (vectorized)
            #probability_on_all_visible_neurons_all_samples = 1 / (1 + np.exp(-weighted_sum_all_samples))
            probability_on_all_visible_neurons_all_samples = sigmoid(weighted_sum_all_samples)

            # Generate random values for each visible neuron of every sample for activation decision
            #random_values = np.random.rand(n_samples, self.ndim_visible)

            # Determine activation based on probability and random values (vectorized)
            #activation_all_visible_neurons_all_samples = (random_values < probability_on_all_visible_neurons_all_samples).astype(int)
            activation_all_visible_neurons_all_samples = sample_categorical(probability_on_all_visible_neurons_all_samples)

            # Return the probabilities and activations for all samples
            return probability_on_all_visible_neurons_all_samples, activation_all_visible_neurons_all_samples


    
    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    

    def untwine_weights(self):
        
        self.weight_v_to_h = np.copy( self.weight_vh )
        self.weight_h_to_v = np.copy( np.transpose(self.weight_vh) )
        self.weight_vh = None

    def get_h_given_v_dir(self,visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]

        # [DONE TASK 4.2] perform same computation as the function 'get_h_given_v' but with directed connections (replace the zeros below) 



        # Matrix-based implementation for all samples
        # Compute the weighted sum for all hidden neurons for all samples using matrix multiplication
        # Each row corresponds to one sample, each column to one hidden neuron
        weighted_sum_all_samples = np.dot(visible_minibatch, self.weight_v_to_h) + self.bias_h

        # Compute the probability of activation for all hidden neurons across all samples (vectorized)
        #probability_on_all_hidden_neurons_all_samples = 1 / (1 + np.exp(-weighted_sum_all_samples))
        probability_on_all_hidden_neurons_all_samples = sigmoid(weighted_sum_all_samples)
        #print("probability_on_all_hidden_neurons_all_samples", probability_on_all_hidden_neurons_all_samples)

        # Generate random values for each hidden neuron of every sample for activation decision
        #random_values = np.random.rand(n_samples, self.ndim_hidden)

        # Determine activation based on probability and random values (vectorized)
        #activation_all_hidden_neurons_all_samples = (random_values < probability_on_all_hidden_neurons_all_samples).astype(int)
        activation_all_hidden_neurons_all_samples = sample_binary(probability_on_all_hidden_neurons_all_samples)

        # Since we already have matrix-based results, no need for appending
        # Directly return the probability and activation matrices
        return probability_on_all_hidden_neurons_all_samples, activation_all_hidden_neurons_all_samples





        
        #return np.zeros((n_samples,self.ndim_hidden)), np.zeros((n_samples,self.ndim_hidden))


    def get_v_given_h_dir(self,hidden_minibatch):


        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_h_to_v is not None
        
        n_samples = hidden_minibatch.shape[0]
        
        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            
            # [DONE TASK 4.2] Note that even though this function performs same computation as 'get_v_given_h' but with directed connections,
            # this case should never be executed : when the RBM is a part of a DBN and is at the top, it will have not have directed connections.
            # Appropriate code here is to raise an error (replace pass below)

            
            #pass

            raise Exception("self.is_top when calling get_v_given_h_dir, SHOULD NEVER HAPPEN!!!")

            
        else:
                        
            # [DONE TASK 4.2] performs same computaton as the function 'get_v_given_h' but with directed connections (replace the pass and zeros below)             

            # Compute weighted sums for all samples and all visible neurons at once using matrix multiplication
            # Each row of the result corresponds to a sample
            #weighted_sum_all_samples = np.dot(hidden_minibatch, self.weight_vh.T) + self.bias_v
            weighted_sum_all_samples = np.dot(hidden_minibatch, self.weight_h_to_v) + self.bias_v

            # Compute the probability of activation for all visible neurons for all samples (vectorized)
            #probability_on_all_visible_neurons_all_samples = 1 / (1 + np.exp(-weighted_sum_all_samples))
            probability_on_all_visible_neurons_all_samples = sigmoid(weighted_sum_all_samples)

            # Generate random values for each visible neuron of every sample for activation decision
            #random_values = np.random.rand(n_samples, self.ndim_visible)

            # Determine activation based on probability and random values (vectorized)
            #activation_all_visible_neurons_all_samples = (random_values < probability_on_all_visible_neurons_all_samples).astype(int)

            activation_all_visible_neurons_all_samples = sample_binary(probability_on_all_visible_neurons_all_samples)



            # Return the probabilities and activations for all samples
            return probability_on_all_visible_neurons_all_samples, activation_all_visible_neurons_all_samples


            #pass
            
        #return np.zeros((n_samples,self.ndim_visible)), np.zeros((n_samples,self.ndim_visible))        
    
    
        
    def update_generate_params(self,inps,trgs,preds):
        
        """Update generative weight "weight_h_to_v" and bias "bias_v"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [DON'T DO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.
        
        self.delta_weight_h_to_v += 0
        self.delta_bias_v += 0
        
        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v 
        
        return
    
    def update_recognize_params(self,inps,trgs,preds):
        
        """Update recognition weight "weight_v_to_h" and bias "bias_h"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [DON'T DO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_v_to_h += 0
        self.delta_bias_h += 0

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h
        
        return    
