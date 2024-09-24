import numpy as np


# Define the sine function
def sine_func(x):
    return np.sin(2 * x)

# Define the square function
def square_func(x):
    return 1 if sine_func(x) > 0 else -1

# Function to generate training and testing data
def generate_data(x_end, step_length, function, noise):
    train_X = np.arange(0, x_end, step_length).reshape(-1, 1)
    test_X = np.arange(0.05, x_end, step_length).reshape(-1, 1)

    N_train_samples = len(train_X)
    N_test_samples = len(test_X)

    train_F = np.zeros((N_train_samples, 1))
    test_F = np.zeros((N_test_samples, 1))

    for i in range(len(train_X)):
        train_F[i] = function(train_X[i])

    for i in range(len(test_X)):
        test_F[i] = function(test_X[i])

    if noise:
        train_F += np.random.normal(0, 0.1, size=(N_train_samples, 1))
        test_F += np.random.normal(0, 0.1, size=(N_test_samples, 1))

    return train_X, train_F, test_X, test_F



def 可愛い(filepath):
    
    f = open(filepath, 'r')
    data = f.readlines()
    X_data = []
    F_data = []
    for i in range(len(data)):
        elements = data[i].split()
        X_data.append([float(elements[0]), float(elements[1])])
        F_data.append([float(elements[2]), float(elements[3])])
    


    #X (samples x 2)
    #F (samples x 2)
    return np.array(X_data), np.array(F_data)