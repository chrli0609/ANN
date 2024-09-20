#task 2-1
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

import copy






class MLP(nn.Module):
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

    def init_weights(self):
        for i in range(len(self.layers)):
            torch.nn.init.normal_(self.layers[i].weight, mean=0.0, std=0.1)

    

        
    def forward(self, x):
        # Pass input through each layer with ReLU activation except the output layer
        for layer in self.layers[:-1]:
            #x = torch.sigmoid(layer(x))
            x = F.relu(layer(x))
        
        # Pass through the final layer without activation
        x = self.layers[-1](x)
        
        return x 



    def train_and_validate(self, inputs, targets, validation_inputs, validation_targets, num_epochs, criterion, optimizer, early_stopping):

        train_loss_list = []
        valid_loss_list = []

        n_train_samples = inputs.shape[-1]
        n_val_samples = validation_inputs.shape[-1]


        MAX_PATIENCE = 14

        
        #Initialize Variables for EarlyStopping
        best_loss = float('inf')
        best_model_weights = None
        patience = MAX_PATIENCE



        for epoch in range(num_epochs):

            
            #Validation
            running_vloss = 0.0
            self.eval()
            with torch.no_grad():
                validation_outputs = self(validation_inputs)
                val_loss = criterion(validation_outputs, validation_targets.t())#/ n_val_samples



            self.train()

            #Reset gradient
            optimizer.zero_grad()


            #Training
            outputs = self(inputs)
            train_loss = criterion(outputs, targets.t()) #/ n_train_samples
            train_loss.backward()
            optimizer.step()


            #print("outputs", outputs.shape)
            #print("targets", targets.t().shape)




                
                #print("val_loss", val_loss)
                #running_vloss += vloss
                #print("val_loss", val_loss)

            #avg_vloss = running_vloss / (i + 1)
            #print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))


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
            #print("Epoch", epoch, "\t: training loss =", train_loss.item(), "\t: validation loss =", val_loss.item())

            train_loss_list.append(train_loss.item())
            valid_loss_list.append(val_loss.item())
    

        print("Finished training\tfinal training error:", train_loss_list[-1], "\tfinal validation error:", valid_loss_list[-1])


        #If no early_Stopping --> just take the last sample as best
        if not early_stopping:
            best_model_weights = copy.deepcopy(self.state_dict())
        

        return best_model_weights, train_loss_list, valid_loss_list


class Dataset:

    def __init__(self, beta, gamma, n, tau, guess_length):
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.tau = tau

        self.guess_length = guess_length




    def eulers_method(self, end_t):
        x = np.zeros(end_t + self.guess_length)
        #x = np.array([None]*(end_t+self.guess_length))
    
        x[0] = 1.5    
        for t in range(end_t):
            x[t + 1] = x[t] + self.beta*x[t - self.tau] / (1 + x[t - self.tau]**self.n) - 0.1 * x[t]

        return x



    #Organise such that:
    #X_in = [[x(301), x(296), x(291), x(286), x(281)],
    #        [x(302), x(297), x(292), x(287), x(282)],
    #        [x(303), x(298), x(293), x(288), x(283)],
    #           ...
    #        [x(1499), x(1494), x(1489), x(1484), x(1479)]
    #        [x(1500), x(1495), x(1490), x(1485), x(1480)]]

    #X_out = [x(306),
    #         x(307),
    #         x(308),
    #         ... 
    #         x(1504),
    #         x(1505)]
    def organise_to_in_out_matrices(self, x, start_t, end_t):
        
        num_samples = (end_t + 1) - start_t 


        X_in = np.ones((num_samples, self.guess_length))
        X_out = np.ones((num_samples, 1))


        row_it = 0
        for i in range(start_t-1, end_t):

            X_out[row_it] = x[i + self.guess_length]

            for j in range(self.guess_length):

                X_in[row_it][j] = x[i-5*j]
            
            row_it += 1

        return X_in, X_out



    #Expected in and out dimensions: (5, 1200) and (1, 1200) respectively
    #split into training(1|5, 800) validation(1|5, 200) testing(1|5, 200)
    #Args are np.array but function returns torch.tensor
    def split_samples(self, X_in, X_out, valid_dist, n_samples):

        X = np.hstack((X_in, X_out))
        np.random.shuffle(X)
        X = X.T

        X_in = X[:-1, :]
        X_out = X[-1, :]
        X_out = X_out.reshape(-1,1)


        
        training_samples = int(n_samples * (1-valid_dist))
        testing_samples = 200
        
        input_training = X_in[:, :training_samples]
        input_validation = X_in[:, training_samples:(n_samples - testing_samples)]
        input_testing = X_in[:, (n_samples - testing_samples):]

        output_training = X_out[:training_samples]
        output_validation = X_out[training_samples:(n_samples - testing_samples)]
        output_testing = X_out[(n_samples - testing_samples):]

        #print("output_training", output_training.shape)
        #print("output_validation", output_validation.shape)
        #print("output_testing", output_testing.shape)
        
        return torch.tensor(input_training.T), torch.tensor(input_validation.T), torch.tensor(input_testing.T), torch.tensor(output_training.T), torch.tensor(output_validation.T), torch.tensor(output_testing.T)
        

        
    def split_samples_no_shuffle_test(self, X_in, X_out, valid_dist, n_samples):
        training_samples = int(n_samples * (1-valid_dist))
        testing_samples = 200


        #Carve out last 200 for testing
        input_testing = X_in[(n_samples-testing_samples):, :]
        output_testing = X_out[(n_samples-testing_samples):]

        X_in = X_in[:(n_samples-testing_samples), :]
        X_out = X_out[:(n_samples-testing_samples)]




        #Stack together and shuffle
        X = np.hstack((X_in, X_out))


        np.random.shuffle(X)
        X = X.T

        X_in = X[:-1, :]
        X_out = X[-1, :]
        X_out = X_out.reshape(-1,1)


        
        
        
        input_training = X_in[:, :training_samples]
        input_validation = X_in[:, training_samples:(n_samples - testing_samples)]
        

        output_training = X_out[:training_samples]
        output_validation = X_out[training_samples:(n_samples - testing_samples)]
        

        #print("output_training", output_training.shape)
        #print("output_validation", output_validation.shape)
        #print("output_testing", output_testing.shape)
        
        return torch.tensor(input_training.T), torch.tensor(input_validation.T), torch.tensor(input_testing.T), torch.tensor(output_training.T), torch.tensor(output_validation.T), torch.tensor(output_testing.T)

    



def main(LAYER_SIZES, LEARNING_RATE, WEIGHT_DECAY, ax1, ax2):
    #NUM_RUNS = 10
    NUM_EPOCHS = 100
    layer_sizes = LAYER_SIZES
    ZERO_MEAN = True
    EARLY_STOPPING = False

    t_start = 301
    t_end = 1500
    valid_dist = 0.25

    n_samples = t_end - t_start + 1

    dataset = Dataset(beta=0.2, gamma=0.1, n=10, tau=25, guess_length=5)
    x = dataset.eulers_method(t_end)

    X_in, X_out = dataset.organise_to_in_out_matrices(x, t_start, t_end)
    num_samples, input_size = X_in.shape
    _, output_size = X_out.shape

    np.random.seed(23)

    run_list_train_loss = []
    run_list_valid_loss = []
    predicted_outputs_list = []
    X_out_list = []

    for i in range(NUM_RUNS):
        
        
        if ZERO_MEAN:
            X_in_run = X_in + np.random.normal(0, 0.15, (num_samples, input_size))
            X_out_run = X_out + np.random.normal(0, 0.15, (n_samples, output_size))

        #input_training, input_validation, input_testing, output_training, output_validation, output_testing = dataset.split_samples_no_shuffle_test(X_in, X_out, valid_dist, n_samples)
        input_training, input_validation, input_testing, output_training, output_validation, output_testing = dataset.split_samples(X_in_run, X_out_run, valid_dist, n_samples)

    
        model = MLP(input_size=input_size, layer_sizes=layer_sizes, output_size=output_size)
        model.init_weights()

        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        best_model_weights, train_loss_list, valid_loss_list = model.train_and_validate(input_training, output_training, input_validation, output_validation, NUM_EPOCHS, criterion, optimizer, early_stopping=EARLY_STOPPING)


        #Append losses to list for calculating mean of them
        run_list_train_loss.append(train_loss_list)
        run_list_valid_loss.append(valid_loss_list)

        predicted_outputs_list.append(model(torch.tensor(X_in_run).double()).detach().numpy())
        X_out_list.append(X_out_run)
    


    #Find mean training and validation error between runs
    mean_train_loss_list = np.mean(run_list_train_loss, axis=0)
    mean_valid_loss_list = np.mean(run_list_valid_loss, axis=0)

    #Find mean predicted outputs for each run
    mean_predicted_outputs_list = np.mean(predicted_outputs_list, axis=0)
    mean_X_out_list = np.mean(X_out_list, axis=0)
    



    #print("run_list_train_loss", run_list_train_loss)
    #print("run_list_valid_loss", run_list_valid_loss)
    #print("mean_train_loss_list", mean_train_loss_list)
    #print("mean_valid_loss_list", mean_valid_loss_list)
        

    # Plot training and validation loss
    ax1.plot(mean_train_loss_list, label='Training Error', color='blue')
    ax1.plot(mean_valid_loss_list, label='Validation Error', color='red')
    ax1.set_title(f"Nodes ({LAYER_SIZES})\nLoss Curve LR={LEARNING_RATE}\nmin(E_val)={round(min(valid_loss_list),3)}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Plot predicted vs actual outputs
    ax2.plot(mean_X_out_list, label='Actual', color='green')
    ax2.plot(mean_predicted_outputs_list, label='Predicted', color='orange')
    ax2.set_title("Actual vs Predicted")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Output")
    ax2.legend(loc="upper right")


# Define the parameter list (layer sizes and learning rates)
NUM_RUNS = 10
param_list = [
    [[4, 3], 0.01],
    [[4, 6], 0.01],
    [[4, 9], 0.01]
    #[[6, 8], 0.01],
    #[[7, 9], 0.01]
]

# Define the range of weight decay values
weight_decay_values = [10**(-i) for i in range(1, 7)]  # From 10^(-1) to 10^(-6)

# Function to update the plots for animation
def update_plot(frame):
    fig.clear()
    axes = []

    # Create a 2x3 grid for displaying the models (3 models, 2 plots each)
    num_models = len(param_list)
    for i in range(num_models):
        axes.append((fig.add_subplot(2, num_models, i + 1), fig.add_subplot(2, num_models, i + 4)))

    wd_idx = frame % len(weight_decay_values)  # Get the index for the weight decay
    weight_decay = weight_decay_values[wd_idx]

    # For each model, call the main function to update its plots
    for i, params in enumerate(param_list):
        ax1, ax2 = axes[i]
        main(params[0], params[1], weight_decay, ax1, ax2)

    fig.suptitle(f"Average over {NUM_RUNS} runs\nWeight Decay = {weight_decay:.1e}")

# Set up the figure and axes for animation
fig = plt.figure(figsize=(18, 10))

# Calculate the total number of frames
total_frames = len(weight_decay_values)

# Create the animation
anim = FuncAnimation(fig, update_plot, frames=total_frames, interval=400, repeat=False)

# Save the animation as a GIF
anim.save('../out/all_models_weight_decay_animation.gif', writer='pillow')

