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



    def train_and_validate(self, inputs, targets, validation_inputs, validation_targets, num_epochs, criterion, optimizer):

        train_loss_list = []
        valid_loss_list = []


        #Initialize Variables for EarlyStopping
        best_loss = float('inf')
        best_model_weights = None
        patience = 10



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

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_weights = copy.deepcopy(self.state_dict())  # Deep copy here      
                patience = 10  # Reset patience counter
            else:
                patience -= 1
                if patience == 0:
                    break

            #print loss
            print("Epoch", epoch, "\t: training loss =", train_loss.item(), "\t: validation loss =", val_loss.item())

            train_loss_list.append(train_loss.item())
            valid_loss_list.append(val_loss.item())
    
        return best_model_weights, train_loss_list, valid_loss_list






        
        







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
        


    



def main():
    np.random.seed(42)


    NUM_EPOCHS = 80
    LEARNING_RATE = 0.1

    #Hidden layers only
    layer_sizes = [30, 30]

    t_start = 301
    t_end = 1500
    valid_dist = 0.2
    

    n_samples = t_end - t_start + 1


    dataset = Dataset(beta=0.2, gamma=0.1, n=10, tau=25, guess_length=5)
    #x is a numpy vector that holds all the values of x(t=0) to x(t=1500)
    x = dataset.eulers_method(t_end)

    X_in, X_out = dataset.organise_to_in_out_matrices(x, t_start, t_end)

    input_training, input_validation, input_testing, output_training, output_validation, output_testing = dataset.split_samples(X_in, X_out, valid_dist, n_samples)

    #print(X_in)
    #print(X_out)

    #print("input_train", input_training.shape)
    #print("output_trainig", output_training.shape)
    #print("input_valid", input_validation.shape)
    #print("output_valid", output_validation.shape)

    num_samples, input_size = X_in.shape
    _, output_size = X_out.shape


    model = MLP(input_size=input_size, layer_sizes=layer_sizes, output_size=output_size)



   

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #print("input_trainng", input_training)
    #print("output_training", output_training)
    best_model_weights, train_loss_list, valid_loss_list = model.train_and_validate(input_training, output_training, input_validation, output_validation, NUM_EPOCHS, criterion, optimizer)

    t = np.arange(1, NUM_EPOCHS+1)
    

    plt.plot(t, train_loss_list, c='blue', label='Training Error')
    plt.plot(t, valid_loss_list, c='red', label='Validation Error')
    plt.legend()
    plt.show()





    #Plot resulting line

    plt.plot(X_out)
    print("torch.tensor(X_in).detach()", torch.tensor(X_in).detach())
    print("torch.tensor(X_in)", torch.tensor(X_in))
    tmp = model(torch.tensor(X_in))
    plt.plot(tmp.detach())
    plt.show()



main()