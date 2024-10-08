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
        
        self.rf = { # receptive-fields. Only applicable when visible layer is input data
            "period" : 5000, # iteration period to visualize
            "grid" : [5,5], # size of the grid
            "ids" : np.random.randint(0,self.ndim_hidden,25) # pick some random hidden units
            }
        
        return

        
    def cd1(self,visible_trainset, n_iterations=10000):
        
        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print ("learning CD1")

        #visible_trainset = visible_trainset[0:100,:]
        
        n_samples = visible_trainset.shape[0]


        if n_iterations * self.batch_size != n_samples:
            print("AN ERROR HAS OCCURREEEEEEDDDD: n_iterations * self.batch_size != n_samples")
        
        #v_states_0 = visible_trainset
        #h_states_0 = [None] * n_samples
        #v_states_1 = [None] * n_samples
        #h_states_1 = [None] * n_samples

        for it in range(n_iterations):

            v_batch_probability_on = visible_trainset[it*self.batch_size:(it+1)*self.batch_size][:]


            v_batch_states_0 = np.zeros((self.batch_size, self.ndim_visible))

            for i in range(self.batch_size):
                for j in range(self.ndim_visible):

                    r = random.random()
                    if r < v_batch_probability_on[i][j]:
                        v_batch_states_0[i][j] = 1
                    else:
                        v_batch_states_0[i][j] = 0
 


            
            
            #Awake
            _, h_batch_states_0 = self.get_h_given_v(v_batch_states_0)
            #for i in range(self.batch_size):
            #    h_states_0[i] = h_batch_states_0
            

            #Asleep
            _, v_batch_states_1 = self.get_v_given_h(h_batch_states_0)
            #for i in range(self.batch_size):
            #    v_states_1[i] = v_batch_states



            #Awake 2
            _, h_batch_states_1 = self.get_h_given_v(v_batch_states_1)
            #for i in range(self.batch_size):
            #    h_states_1[i] = h_batch_states_1


            
            #self.update_params(v_states_0, h_states_0, v_states_1, h_states_1)
            self.update_params(v_batch_states_0, h_batch_states_0, v_batch_states_1, h_batch_states_1, n_samples)

            
            
            

	    # [TODO TASK 4.1] run k=1 alternating Gibbs sampling : v_0 -> h_0 ->  v_1 -> h_1.
            # you may need to use the inference functions 'get_h_given_v' and 'get_v_given_h'.
            # note that inference methods returns both probabilities and activations (samples from probablities) and you may have to decide when to use what.

            # [TODO TASK 4.1] update the parameters using function 'update_params'
            
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

        # [TODO TASK 4.1] get the gradients from the arguments (replace the 0s below) and update the weight and bias parameters



        #print("Updating params")
        #print("v_0\n", v_0.tolist(), "\n", np.sum(v_0))
        #print("h_0\n", h_0, "\n", np.sum(h_0))
        #print("v_k\n", v_k, "\n", np.sum(v_k))
        #print("h_k\n", h_k, "\n", np.sum(h_k))

        delta_bias_v = np.zeros(self.ndim_visible)
        delta_bias_h = np.zeros(self.ndim_hidden)
        
        delta_weight = np.zeros((v_0.shape[1], h_0.shape[1]))

        # Update rule for the weights based on Contrastive Divergence
        # Calculate the outer product for each sample and average over all samples
        for n in range(self.batch_size):

            #Bias
            delta_bias_v += (v_0[n] - v_k[n]) / n_samples
            delta_bias_h += (h_0[n] - h_k[n]) / n_samples
                

            positive_grad = np.outer(v_0[n], h_0[n])  # Data-dependent term
            negative_grad = np.outer(v_k[n], h_k[n])  # Model-dependent term


            delta_weight += (positive_grad - negative_grad) / n_samples


        

        
        self.delta_bias_v += delta_bias_v
        self.delta_weight_vh += delta_weight
        self.delta_bias_h += delta_bias_h
        
        self.bias_v += self.delta_bias_v
        self.weight_vh += self.delta_weight_vh
        self.bias_h += self.delta_bias_h
        
        return

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


        # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of hidden layer (replace the zeros below) 

        


        probability_of_activation_all_neurons_of_all_samples = []
        activation_of_all_neurons_of_all_samples = []
        for k in range(n_samples):
            

            probability_on_all_hidden_neurons = [None] * self.ndim_hidden
            activation_all_hidden_neurons = [None] * self.ndim_hidden
            for j in range(self.ndim_hidden):
                

                #Compute contributions from all neighboring nodes
                weighted_sum = 0
                for i in range(self.ndim_visible):
                    weighted_sum += visible_minibatch[k][i] * self.weight_vh[i][j]
                
                #Compute probability of hidden neurons j is +1
                probability_on_all_hidden_neurons[j] = 1 / (1 + np.exp(-self.bias_h[j] - weighted_sum))


                #Throw dice to decide if state should be +1 or -1
                r = random.random()
                
                #print("probability_on_all_hidden_neurons[j]", np.array(probability_on_all_hidden_neurons[j]).shape)
                if r < probability_on_all_hidden_neurons[j]:
                    activation_all_hidden_neurons[j] = 1
                else:
                    activation_all_hidden_neurons[j] = 0
            
            

            #Append the probability of hidden neurons = 1
            probability_of_activation_all_neurons_of_all_samples.append(probability_on_all_hidden_neurons)
            activation_of_all_neurons_of_all_samples.append(activation_all_hidden_neurons)
                
        

        return np.array(probability_of_activation_all_neurons_of_all_samples), np.array(activation_of_all_neurons_of_all_samples)
        #return np.zeros((n_samples,self.ndim_hidden)), np.zeros((n_samples,self.ndim_hidden))




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

            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass below). \
            # Note that this section can also be postponed until TASK 4.2, since in this task, stand-alone RBMs do not contain labels in visible layer.
            
            pass
            
        else:
                        
            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass and zeros below)             

            probability_of_activation_all_neurons_of_all_samples = []
            activation_of_all_neurons_of_all_samples = []
            for k in range(n_samples):
                

                probability_on_all_visible_neurons = [None] * self.ndim_visible
                activation_all_visible_neurons = [None] * self.ndim_visible
                for i in range(self.ndim_visible):
                    

                    #print("self.weight_vh", self.weight_vh)
                    #print("hidden_minibatch", hidden_minibatch)
                    
                    #Compute contributions from all neighboring nodes
                    weighted_sum = 0
                    for j in range(self.ndim_hidden):
                        weighted_sum += self.weight_vh[i][j] * hidden_minibatch[k][j]
                    
                    #Compute probability of hidden neurons j is +1
                    probability_on_all_visible_neurons[i] = 1 / (1 + np.exp(-self.bias_v[i] - weighted_sum))


                    #Throw dice to decide if state should be +1 or -1
                    r = random.random()

                    if r < probability_on_all_visible_neurons[i]:
                        activation_all_visible_neurons[i] = 1
                    else:
                        activation_all_visible_neurons[i] = 0
                
                

                #Append the probability of hidden neurons = 1
                probability_of_activation_all_neurons_of_all_samples.append(probability_on_all_visible_neurons)
                activation_of_all_neurons_of_all_samples.append(activation_all_visible_neurons)
                
        

        return np.array(probability_of_activation_all_neurons_of_all_samples), np.array(activation_of_all_neurons_of_all_samples)
        
        #return np.zeros((n_samples,self.ndim_visible)), np.zeros((n_samples,self.ndim_visible))


    
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

        # [TODO TASK 4.2] perform same computation as the function 'get_h_given_v' but with directed connections (replace the zeros below) 
        
        return np.zeros((n_samples,self.ndim_hidden)), np.zeros((n_samples,self.ndim_hidden))


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
            
            # [TODO TASK 4.2] Note that even though this function performs same computation as 'get_v_given_h' but with directed connections,
            # this case should never be executed : when the RBM is a part of a DBN and is at the top, it will have not have directed connections.
            # Appropriate code here is to raise an error (replace pass below)
            
            pass
            
        else:
                        
            # [TODO TASK 4.2] performs same computaton as the function 'get_v_given_h' but with directed connections (replace the pass and zeros below)             

            pass
            
        return np.zeros((n_samples,self.ndim_visible)), np.zeros((n_samples,self.ndim_visible))        
    
    
        
    def update_generate_params(self,inps,trgs,preds):
        
        """Update generative weight "weight_h_to_v" and bias "bias_v"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.
        
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

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_v_to_h += 0
        self.delta_bias_h += 0

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h
        
        return    
