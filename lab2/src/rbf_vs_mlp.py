import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


from gen_data_func import *

# Define the MLP class
class MLP(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(MLP, self).__init__()
        
        # Create a list of layers
        layers = []
        # Input to first hidden layer
        layers.append(nn.Linear(input_size, layer_sizes[0]))
        
        # Hidden layers
        for i in range(1, len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        
        # Output layer
        layers.append(nn.Linear(layer_sizes[-1], output_size))
        
        # Save layers as a ModuleList
        self.layers = nn.ModuleList(layers)

        self.double()

    def forward(self, x):
        # Pass input through each layer and apply ReLU activation for hidden layers
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        # Pass through the final layer without activation for regression
        x = self.layers[-1](x)
        return x

# Generate sine(2x) data with noise
def generate_sine_data(num_samples=1000, noise=True):
    x = np.linspace(-np.pi, np.pi, num_samples).reshape(-1, 1)  # Input features
    y = np.sin(2 * x)  # Target values
    if noise:
        y += np.random.normal(0, 0.1, y.shape)  # Add some noise to the data
    return x, y

# Prepare data for training and testing
def prepare_data(train_ratio=0.8):
    x, y = generate_sine_data()
    num_train = int(train_ratio * len(x))

    # Convert data to tensors
    train_x = torch.tensor(x[:num_train], dtype=torch.float32)
    train_y = torch.tensor(y[:num_train], dtype=torch.float32)
    test_x = torch.tensor(x[num_train:], dtype=torch.float32)
    test_y = torch.tensor(y[num_train:], dtype=torch.float32)

    return train_x, train_y, test_x, test_y

# Train the MLP model
def train_mlp(model, train_loader, optimizer, criterion, num_epochs):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.6f}")
    
    return train_losses

# Test the MLP model
def test_mlp(model, test_x, test_y):
    model.eval()
    with torch.no_grad():
        predictions = model(test_x)
    return predictions

# Hyperparameters and settings
input_size = 1
layer_sizes = [64]
output_size = 1
learning_rate = 0.01
num_epochs = 100
batch_size = 32



STEP_LENGTH = 0.1
NOISE = True
#DATA_FUNC = sine_func
DATA_FUNC = square_func

# Prepare training and testing data
#train_x, train_y, test_x, test_y = prepare_data()
#train_x, train_y, test_x, test_y = generate_data(2*np.pi, data_steplength, )
train_x, train_y, test_x, test_y = generate_data(2 * np.pi, STEP_LENGTH, DATA_FUNC, NOISE)

train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y)
test_x = torch.tensor(test_x)
test_y = torch.tensor(test_y)

# Create a DataLoader for batch processing
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = MLP(input_size, layer_sizes, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_losses = train_mlp(model, train_loader, optimizer, criterion, num_epochs)

# Test the model
predictions = test_mlp(model, test_x, test_y)

# Plot training losses
plt.plot(train_losses, label="Training Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.legend()
plt.show()

# Plot the sine wave and model predictions
plt.plot(test_x.numpy(), test_y.numpy(), label='True Function')
plt.plot(test_x.numpy(), predictions.numpy(), label='Predicted Function')
plt.xlabel('x')
plt.ylabel('sin(2x)')
plt.legend()
plt.title('MLP Regression on sin(2x)')
plt.show()
