import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import copy


import torch
import torch.nn as nn
import torch.nn.functional as F



from model import MLP
from functions import *

np.random.seed(2345)



#Output folder
OUT_FOLDER = "../out/task_1-2/"

# Data and parameters
NUM_SAMPLES_PER_CLASS = 100
mA = np.array([1.0, 0.3])
sigmaA = 0.2
mB = np.array([0.0, -0.1])
sigmaB = 0.3

# Generate random data
IN_DIM = 2
NUM_EPOCHS = 50
LEARNING_RATE = 0.01
MAX_HIDDEN_NODES_TO_TRY = 30



################################################################################################3
###### CHOOSE SUBSAMPLING METHOD #####
subsampling_method = subsampling_25_from_each_class
#subsampling_method = subsampling_50_from_classA
#subsampling_method = subsampling_point_2_lt_0_and_point_8_gt_0_from_A

##### CHOOSE BATCH OR SEQUENTIAL GRADIENT DESCENT #####
is_batch_learning = False

################################################################################################



















class MLP_torch(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        """
        Initializes the MLP.
        
        Args:
        - input_size (int): Size of the input layer.
        - layer_sizes (list of int): A list of integers where each element specifies the number of neurons in a hidden layer.
        - output_size (int): Size of the output layer.
        """
        super(MLP, self).__init__()
        
        # Create a list to hold all layers
        layers = []
        
        # Input layer to the first hidden layer
        layers.append(nn.Linear(input_size, layer_sizes[0]))
        
        # Add hidden layers
        for i in range(1, len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        
        # Output layer
        layers.append(nn.Linear(layer_sizes[-1], output_size))
        
        # Save the layers as a ModuleList so PyTorch can track them
        self.layers = nn.ModuleList(layers)


        self.double()
        
    def forward(self, x):
        # Pass input through each layer with ReLU activation except the output layer
        for layer in self.layers[:-1]:
            x = torch.sigmoid(layer(x)-1)
            #x = F.relu(layer(x))
        

        x = self.layers[-1](x)
        
        return x 



    def train_and_validate(self, inputs, targets, validation_inputs, validation_targets, num_epochs, criterion, optimizer, early_stopping):

        train_loss_list = []
        valid_loss_list = []
        MAX_PATIENCE = 14

        
        #Initialize Variables for EarlyStopping
        best_loss = float('inf')
        best_model_weights = None
        patience = MAX_PATIENCE



        for epoch in range(num_epochs):

            self.train()

            #Reset gradient
            optimizer.zero_grad()


            #Training
            outputs = self(inputs)
            train_loss = criterion(outputs, targets.t())
            train_loss.backward()
            optimizer.step()


            #print("outputs", outputs.shape)
            #print("targets", targets.t().shape)



            #Validation
            
            self.eval()
            with torch.no_grad():
                validation_outputs = self(validation_inputs)
                val_loss = criterion(validation_outputs, validation_targets.t())

            #print("validation_outputs", validation_outputs.shape)
            #print("validation_targets", validation_targets.t().shape)

            if early_stopping:
                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_weights = copy.deepcopy(self.state_dict())  # Deep copy here
                    best_model_epoch = epoch
                    patience = MAX_PATIENCE  # Reset patience counter
                else:
                    patience -= 1
                    if patience == 0:
                        print("Early Stopping at epoch:", epoch, " saved model is from epoch:", best_model_epoch)
                        break

            #print loss
            print("Epoch", epoch, "\t: training loss =", train_loss.item(), "\t: validation loss =", val_loss.item())

            train_loss_list.append(train_loss.item())
            valid_loss_list.append(val_loss.item())
    

        #If no early_Stopping --> just take the last sample as best
        if not early_stopping:
            best_model_weights = copy.deepcopy(self.state_dict()) 

        return best_model_weights, train_loss_list, valid_loss_list









# Generate data and weights
X, T, X_test, T_test, color_list, color_list_test = generate_random_non_linear_input_and_weights(
    IN_DIM, NUM_SAMPLES_PER_CLASS, mA, mB, sigmaA, sigmaB, subsampling_method
)


#plot_data(X, color_list, X_test, color_list_test)



#print("X", X)
#print("T", T)


#print("X_text", X_test)
#print("T_text", T_test)

# Generate list of number of hidden nodes to try
hidden_nodes_list = np.arange(1, MAX_HIDDEN_NODES_TO_TRY + 1, 1)

# Arrays to store errors for each neuron configuration
train_error_hidden_nodes_list = []
valid_error_hidden_nodes_list = []



# Train and record errors for each neuron configuration
for hidden_nodes in hidden_nodes_list:

    '''
    model = MLP_torch(IN_DIM, [5], 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    _, train_error, valid_error = model.train_and_validate(torch.tensor(X.T), torch.tensor(T), torch.tensor(X_test.T), torch.tensor(T_test), NUM_EPOCHS, criterion, optimizer, False)
    
    
    '''
    model = MLP(IN_DIM, hidden_nodes, 1, LEARNING_RATE)
    print("NOW EXPLORING", hidden_nodes, "neurons in hidden layer")
    if is_batch_learning:
        train_error, valid_error = model.training_w_valid(X, T, X_test, T_test, NUM_EPOCHS)
    else:
        train_error, valid_error = model.training_sequential_w_valid(X, T, X_test, T_test, NUM_EPOCHS)
    
    

    train_error_hidden_nodes_list.append(train_error)
    valid_error_hidden_nodes_list.append(valid_error)



    o_valid, _ = model.forward_pass(X_test)
    print("o_valid", o_valid)
    class_o = decide_class(o_valid[0])
    print("class_o", class_o)
    final_color_list = generate_color_list(class_o)
    print("final_color_list", final_color_list)


    fig, ax = plt.subplots(nrows=2, ncols=1)
    plt.subplot(2, 1, 1)
    plt.title("Validation Data")
    plt.scatter(X_test[0,:], X_test[1,:], c=color_list_test)

    plt.subplot(2, 1, 2)
    plt.title("Predictions")
    plt.scatter(X_test[0,:], X_test[1,:], c=final_color_list)
    #plt.show()
    plt.savefig("../out/task_1-4/"+str(hidden_nodes) + "_hidden_nodes_pred_scatter.png")
	#plt.legend(loc="upper right")



    



#print("train_error_hidden_nodes_list", train_error_hidden_nodes_list)
#print("valid_error_hidden_nodes_list", valid_error_hidden_nodes_list)



# Call the animation function when needed
animate_training_validation_errors(NUM_EPOCHS,
                                hidden_nodes_list,
                                train_error_hidden_nodes_list,
                                valid_error_hidden_nodes_list,
                                subsampling_method,
                                is_batch_learning,
                                LEARNING_RATE,
                                OUT_FOLDER
                                )


