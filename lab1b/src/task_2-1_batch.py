import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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

        # Convert model to double precision
        self.double()
        
    def forward(self, x):
        # Pass input through each layer with Sigmoid activation for hidden layers, except the output layer
        for layer in self.layers[:-1]:
            x = torch.sigmoid(layer(x))
        
        # Pass through the final layer without activation (linear output)
        x = self.layers[-1](x)
        
        return x

    def train_and_validate(self, train_loader, validation_loader, num_epochs, criterion, optimizer):
        train_loss_list = []
        valid_loss_list = []

        best_loss = float('inf')
        best_model_weights = None
        patience = 10

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            epoch_train_loss = 0

            for batch_inputs, batch_targets in train_loader:
                # Reset gradients
                optimizer.zero_grad()

                # Forward pass: compute predicted outputs
                outputs = self.forward(batch_inputs)

                # Compute loss
                loss = criterion(outputs, batch_targets)
                # Backward pass: compute gradient
                loss.backward()

                # Optimize
                optimizer.step()

                epoch_train_loss += loss.item()

  

            # Validation
            self.eval()  # Set the model to evaluation mode
            epoch_valid_loss = 0

            with torch.no_grad():  # Disable gradient computation during validation
                for val_inputs, val_targets in validation_loader:
                    val_outputs = self.forward(val_inputs)
                    val_loss = criterion(val_outputs, val_targets)
                    epoch_valid_loss += val_loss.item()
            #Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_weights = copy.deepcopy(self.state_dict())  # Deep copy here      
                patience = 10  # Reset patience counter
            else:
                patience -= 1
                if patience == 0:
                    break



            # Compute average training loss for the epoch
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_loss_list.append(avg_train_loss)

            # Compute average validation loss for the epoch
            avg_valid_loss = epoch_valid_loss / len(validation_loader)
            valid_loss_list.append(avg_valid_loss)

            # Print losses
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}")

        return train_loss_list, valid_loss_list


class Dataset:
    def __init__(self, beta, gamma, n, tau, guess_length):
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.tau = tau
        self.guess_length = guess_length

    def eulers_method(self, end_t):
        x = np.empty(end_t + self.guess_length)
        x[0] = 1.5    
        for t in range(end_t):
            x[t + 1] = x[t] + self.beta * x[t - self.tau] / (1 + x[t - self.tau] ** self.n) - 0.1 * x[t]
        return x

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

    def split_samples(self, X_in, X_out, valid_dist, n_samples):
        X = np.hstack((X_in, X_out))
        np.random.shuffle(X)
        X = X.T
        X_in = X[:-1, :]
        X_out = X[-1, :]

        training_samples = int(n_samples * (1-valid_dist))
        testing_samples = 200
        
        input_training = X_in[:, :training_samples]
        input_validation = X_in[:, training_samples:(n_samples - testing_samples)]
        input_testing = X_in[:, (n_samples - testing_samples):]

        output_training = X_out[:training_samples]
        output_validation = X_out[training_samples:(n_samples - testing_samples)]
        output_testing = X_out[(n_samples - testing_samples):]
        
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



def main():
    t_start = 301
    t_end = 1500
    n_samples = t_end - t_start + 1

    NUM_EPOCHS = 80
    LEARNING_RATE = 0.01
    BATCH_SIZE = 1

    valid_dist = 0.5

    

    dataset = Dataset(beta=0.2, gamma=0.1, n=10, tau=25, guess_length=5)
    x = dataset.eulers_method(t_end)

    X_in, X_out = dataset.organise_to_in_out_matrices(x, t_start, t_end)
    input_training, input_validation, input_testing, output_training, output_validation, output_testing = dataset.split_samples(X_in, X_out, valid_dist, n_samples)
    #input_training, input_validation, input_testing, output_training, output_validation, output_testing = dataset.split_samples_no_shuffle_test(X_in, X_out, valid_dist, n_samples)

    num_samples, input_size = X_in.shape
    _, output_size = X_out.shape

    model = MLP(input_size=input_size, layer_sizes=[4, 4], output_size=output_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Convert data to TensorDataset for DataLoader
    #print("input_training", input_training.shape)
    #print("output_training", output_training.shape)
    train_dataset = TensorDataset(input_training, output_training)
    validation_dataset = TensorDataset(input_validation, output_validation)

    # DataLoader for batch learning
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("train_loader", train_loader)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_loss_list, valid_loss_list = model.train_and_validate(train_loader, validation_loader, NUM_EPOCHS, criterion, optimizer)
    
    t = np.arange(1, len(train_loss_list)+1)

    # Plot losses
    plt.plot(t, train_loss_list, c='blue', label='Training Loss')
    plt.plot(t, valid_loss_list, c='red', label='Validation Loss')
    plt.legend()
    plt.show()
    
    plt.plot(X_out)
    tmp = model(torch.tensor(X_in))
    plt.plot(tmp.detach())
    plt.show()

main()

