#task 2-1
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

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
    NUM_EPOCHS = 400
    layer_sizes = LAYER_SIZES
    ZERO_MEAN = True
    EARLY_STOPPING = True

    t_start = 301
    t_end = 1500
    valid_dist = 0.25

    n_samples = t_end - t_start + 1

    dataset = Dataset(beta=0.2, gamma=0.1, n=10, tau=25, guess_length=5)
    x = dataset.eulers_method(t_end)

    X_in, X_out = dataset.organise_to_in_out_matrices(x, t_start, t_end)
    num_samples, input_size = X_in.shape
    _, output_size = X_out.shape


    if ZERO_MEAN:
        X_in += np.random.normal(0, 0.05, (num_samples, input_size))
        X_out += np.random.normal(0, 0.05, (n_samples, output_size))

    #input_training, input_validation, input_testing, output_training, output_validation, output_testing = dataset.split_samples(X_in, X_out, valid_dist, n_samples)
    input_training, input_validation, input_testing, output_training, output_validation, output_testing = dataset.split_samples_no_shuffle_test(X_in, X_out, valid_dist, n_samples)

    

    model = MLP(input_size=input_size, layer_sizes=layer_sizes, output_size=output_size)

    criterion = nn.MSELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_model_weights, train_loss_list, valid_loss_list = model.train_and_validate(input_training, output_training, input_validation, output_validation, NUM_EPOCHS, criterion, optimizer, early_stopping=EARLY_STOPPING)

    # Plot training and validation loss
    ax1.plot(train_loss_list, label='Training Error', color='blue')
    ax1.plot(valid_loss_list, label='Validation Error', color='red')
    ax1.set_title(f"Nodes ({LAYER_SIZES}) Loss Curve (LR={LEARNING_RATE}, WD={WEIGHT_DECAY}), min(E_val)={round(min(valid_loss_list),3)}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Plot predicted vs actual outputs
    ax2.plot(X_out, label='Actual', color='green')
    predicted_outputs = model(torch.tensor(X_in).double()).detach().numpy()
    ax2.plot(predicted_outputs, label='Predicted', color='orange')
    ax2.set_title("Actual vs Predicted")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Output")
    ax2.legend(loc="upper right")

param_list = [
    [[3, 2], 0.01, 10**(-2)],
    [[4, 4], 0.01, 10**(-2)],
    [[5, 6], 0.01, 10**(-2)]
]


np.random.seed(1050)

# Create subplots
fig, axes = plt.subplots(nrows=len(param_list), ncols=2, figsize=(12, 8))

for i, params in enumerate(param_list):
    ax1, ax2 = axes[i]
    main(params[0], params[1], params[2], ax1, ax2)

plt.tight_layout()
#plt.show()

#plt.savefig('../out/task_2-1/compare_3_models_no_noise.png')
plt.savefig('../out/task_2-2/compare_3_models_w_noise.png')


