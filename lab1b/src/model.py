import math
import numpy as np


from functions import *



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

        self.dw = np.zeros((IN_DIM+1, NUM_IN_NODES))
        self.dv = np.zeros((NUM_IN_NODES+1, OUT_DIM))

        self.eta = LEARNING_RATE

        self.alpha = 0.9






    #Batch Backpropogation
    def training(self, X, T, NUM_EPOCHS):
        for epoch in range(NUM_EPOCHS):
            O, H = self.forward_pass(X)


            delta_o, delta_h = self.backward_pass(X, O, H, T)

            self.weight_update(X, H, delta_o, delta_h)


    #Sequential gradient descent while keeping track of erro
    def training_sequential_w_valid(self, X, T, X_test, T_test, NUM_EPOCHS):
        train_error_list = []
        valid_error_list = []

        num_samples = X.shape[1]  # Number of samples

        for epoch in range(NUM_EPOCHS):
            # Shuffle the data at the start of each epoch
            indices = np.random.permutation(num_samples)
            X_shuffled = X[:, indices]
            T_shuffled = T[indices]

            for i in range(num_samples):  # Loop over each training sample sequentially
                x_sample = X_shuffled[:, i:i+1]  # Input vector of size (in_dim, 1)
                t_sample = T_shuffled[i:i+1]  # Target vector of size (out_dim, 1)

                # Forward pass for this single sample
                o_sample, h_sample = self.forward_pass(x_sample)    

                # Backward pass for this single sample
                delta_o_sample, delta_h_sample = self.backward_pass(x_sample, o_sample, h_sample, t_sample) 

                # Update weights based on this sample
                self.weight_update(x_sample, h_sample, delta_o_sample, delta_h_sample)  

            # After processing all training samples for this epoch, compute training error
            o_train, _ = self.forward_pass(X)
            train_mse_loss = mse_loss(o_train, T)  # Mean Squared Error for training set
            train_error_list.append(train_mse_loss) 

            # Compute validation error
            o_valid, _ = self.forward_pass(X_test)
            valid_mse_loss = mse_loss(o_valid, T_test)  # Mean Squared Error for validation set
            valid_error_list.append(valid_mse_loss)

            print(f"Epoch: {epoch}\tTraining Error: {train_mse_loss},\tValidation Error: {valid_mse_loss}")

        return train_error_list, valid_error_list




    #Batch Backpropogation while keeping track of validation error
    def training_w_valid(self, X, T, X_test, T_test, NUM_EPOCHS):
        valid_error_list = []
        train_error_list = []
        for epoch in range(NUM_EPOCHS):
            O, H = self.forward_pass(X)

            delta_o, delta_h = self.backward_pass(X, O, H, T)

            self.weight_update(X, H, delta_o, delta_h)



            #Get training error
            #O_train, _ = self.forward_pass(X)
            #train_error_list.append(mse_loss(O_train, T))
            train_error_list.append(mse_loss(O, T))


            #Get validation error
            O_valid, _ = self.forward_pass(X_test)

            valid_error_list.append(mse_loss(O_valid, T_test))


            print(f"Epoch: {epoch}\tTraining Error: {train_error_list[-1]},\tValidation Error: {valid_error_list[-1]}")

        

        return train_error_list, valid_error_list





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

        # (1 x n)
        delta_o = np.multiply((out-targets), d_activation_func(out))

        # (3 x 1) x (1 x n) = (3 x n)
        delta_h = np.multiply(np.matmul(self.V, delta_o), d_activation_func(hout_bias))
        
        #Remove extra row that was added to handle bias
        delta_h = delta_h[:-1, :]


        #      (1 x n)   (2 x n)
        return delta_o, delta_h




    def weight_update(self, X, H, delta_o, delta_h):
	    _, num_input_samples = X.shape

	    # Add bias terms to input (X) and hidden layer output (H)
	    X_bias = np.concatenate((X, np.ones((1, num_input_samples))), axis=0)
	    H_bias = np.concatenate((H, np.ones((1, num_input_samples))), axis=0)


	    # Update weights
	    self.W -= self.eta * np.matmul(X_bias, delta_h.T)
	    self.V -= self.eta * np.matmul(H_bias, delta_o.T)



	    # Momentum-based weight updates for W and V
	    # Update for W: We use delta_h and input X_bias
	    #self.dw = (self.dw * self.alpha) - np.matmul(X_bias, delta_h.T) * (1 - self.alpha)
	    #self.W += self.dw * self.eta

	    # Update for V: We use delta_o and hidden layer H_bias
	    #self.dv = (self.dv * self.alpha) - np.matmul(H_bias, delta_o.T) * (1 - self.alpha)
	    #self.V += self.dv * self.eta

        






'''
    def weight_update(self, X, H, delta_o, delta_h):
        _, num_input_samples = X.shape

 
        #Fill
        #X_bias = np.concatenate((X, np.ones((1, num_input_samples))), axis=0)
        #H_bias = np.concatenate((H, np.ones((1, num_input_samples))), axis=0)
        H_bias = H
        X_bias = X

        #X:         (3 x n)
        #delta_h:   (2 x n)

        #H:         ()

        print("W", self.W.shape)
        print("V", self.V.shape)
        print("X_bias", X_bias.shape)
        print("delta_h", delta_h.shape)
        print("H_bias", H_bias.shape)
        print("delta_o", delta_o.shape)
        
        

        self.W += -self.eta * np.matmul(X_bias, delta_h.T)
        self.V += -self.eta * np.matmul(H_bias, delta_o.T)


        pat = X_bias
        hout = H_bias

        
        
        print("dw", self.dw)        #(3, 2)
        print("alpha", self.alpha)  #scalar
        print("delta_h", delta_h.shape)   #(2, n)
        print("pat", pat.shape)           #(3, n)
        print("delta_o", delta_o.shape)
        print("hout", hout.shape)
        print("dv", self.dv.shape)
        print("np.matmul(delta_h, pat.T) * (1 - self.alpha)\n", np.matmul(delta_h, pat.T).T * (1 - self.alpha))
        print("(self.dw * self.alpha)\n", (self.dw * self.alpha))
        #self.dw = ((self.dw * self.alpha) - np.matmul(delta_h, pat.T).T * (1 - self.alpha))
        #self.dv = ((self.dv * self.alpha) - np.matmul(delta_o, hout.T).T * (1 - self.alpha))

        print("W", self.W.shape)
        print("dw", self.dw.shape)
        print("eta", self.eta)


        #self.W += self.dw * self.eta
        #self.V += self.dv * self.eta
'''
    
