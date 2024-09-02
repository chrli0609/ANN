import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math


import copy


OUT_FOLDER = "./scatter_vs_lr/"
OFFSET = 0.2

#is_delta_rule = False

train_test_dist = 0.8
NUM_EPOCHS = 20
IN_DIM = 2

#Num samples per class
NUM_SAMPLES_PER_CLASS = 100
CONVERGENCE_FACTOR = 0.5




def lr_list_log(min_lr, max_lr):

    lr_samples = int(math.log10(int(max_lr/min_lr)))

    #Generate list of learning rates to test
    lr_list = [min_lr]
    for i in range(lr_samples):
        curr_value = lr_list[-1]
        lr_list.append(curr_value*10)

    print("lr_list", lr_list)
    return lr_list




def generate_random_input_and_weights(in_dim, n):
    

    mA = [1.0, 0.5]
    mB = [-1.0, 0.0]
    
    sigmaA = 0.5
    sigmaB = 0.5
    
    
    
    #Input data (yet to handle bias)
    classA_row1 = np.random.rand(1, n) * sigmaA + mA[0]
    classA_row2 = np.random.rand(1, n) * sigmaA + mA[1]
    
    classB_row1 = np.random.rand(1, n) * sigmaB + mB[0]
    classB_row2 = np.random.rand(1, n) * sigmaB + mB[1]
    
    
    classA = np.concatenate((classA_row1, classA_row2), axis=0)
    classB = np.concatenate((classB_row1, classB_row2), axis=0)
    
    
    
    
    #Add an extra row in bottom of ones to handle bias
    bias_row = np.ones((1, 2*n))
    
    
    data = np.concatenate((classA, classB), axis=1)
    
    X = np.concatenate((data, bias_row), axis=0) 
    
    #print("X", X)
    
    
    
    
    #Target matrix
    t1 = np.ones(n)
    t2 =-np.ones(n)
    T = np.concatenate((t1, t2))
    
    
    
    
    #Split some parts of the data for testing
    split_index = int(train_test_dist * (2*n))
    
    
    #Merge both answer and input into the same matrix
    all_data = np.concatenate((X, T[:, None].T), axis=0)
    
    
    #Shuffle the data
    np.random.shuffle(np.transpose(all_data))
    
    #print("split_index", split_index)
    
    #Split into training and testing set
    training_set, test_set = all_data[:, :split_index], all_data[:,split_index:]
    
    
    #print("all_data[:,:split_index]", all_data[:,:split_index].shape)
    #print("all_data[:,split_index:]", all_data[:,split_index:].shape)
    
    
    
    #Test input and test target
    X_test = test_set[:in_dim+1,:]
    T_test = test_set[-1,:]
    
    
    
    
    ####### Generate the input X and target T to be used during training #######
    X = training_set[:in_dim+1,:]
    T = training_set[-1, :]
    


    #Generate color list
    color_list = generate_color_list(T)
    
    
    #Add extra row at bottom for bias
    W = np.random.rand(in_dim+1, 1)
    
    
    
    
    return W, X, T, X_test, T_test, color_list


def single_layer_perceptron(W, X, T, LEARNING_RATE, color_list, training_type):

    if training_type == "delta_training":
        plot_title = "Delta Rule Single Layer Perceptron with Learning Rate "
        filepath = "SLP_Delta_rule_lr_"
    elif training_type == "delta_batch_training":
        plot_title = "Batch Delta Rule Single Layer Perceptron with Learning Rate "
        filepath = "SLP_Batch_Delta_rule_lr_"
    elif training_type == "perceptron_learning":
        plot_title = "Perceptron Learning Single Layer Perceptrion with Learning Rate "
        filepath = "SLP_Perceptron_learning_lr_"
    elif training_type == "perceptron_batch_learning":
        plot_title = "Perceptron Batch Learning Single Layer Perceptrion with Learning Rate "
        filepath = "SLP_Perceptron_batch_learning_lr_"
    elif training_type == "sequential_delta_training_convergence":
        plot_title = "Sequential Delta training until convergence with Learning Rate "
        filepath = "SLP_seq_delta_training_until_convergence_learning_lr_"
    elif training_type == "batch_delta_training_convergence":
        plot_title = "Batch Delta training until convergence with Learning Rate "
        filepath = "SLP_batch_delta_training_until_convergence_learning_lr_"
    else:
        print("Error: Training Type Not Found")

    plot_title_str = plot_title + str(LEARNING_RATE)
    
    filepath_str = filepath + str(LEARNING_RATE)
    


    
    

    #Define plotting figure
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot()
    ax.set(xlim=(min(X[0])-OFFSET, max(X[0])+OFFSET), ylim=(min(X[1])-OFFSET, max(X[1])+OFFSET))
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    plt.title(plot_title_str)
    artists = []
    

    
    if training_type == "delta_training":
        W = delta_training(W, X, T, LEARNING_RATE, fig, ax, artists, color_list)
    elif training_type == "batch_delta_training":
        W = batch_delta_training(W, X, T, LEARNING_RATE, fig, ax, artists, color_list)
    elif training_type == "perceptron_learning":
        W = perceptron_training(W, X, T, LEARNING_RATE, fig, ax, artists, color_list)
    elif training_type == "sequential_delta_training_convergence":
        W, num_epochs = sequential_delta_training_until_converge(W, X, T, LEARNING_RATE, fig, ax, artists, color_list)
        print("Artists seq", artists)
    elif training_type == "batch_delta_training_convergence":
        W, num_epochs = batch_delta_training_until_convergence(W, X, T, LEARNING_RATE, fig, ax, artists, color_list)
        print("Artists batch", artists)
    else:
        print("Error: Training Rule Not Found")
    
    
    
    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=800)
    plt.show(block=False)
    
    
    
    
    #Final Weights
    print("Final Weights:\n", W)


    f = OUT_FOLDER + filepath_str + '.gif'
    writergif = animation.PillowWriter(fps=5)
    ani.save(f, writer=writergif)


    if "convergence" in training_type:
        return num_epochs


def sequential_delta_training_until_converge(W, X, T, LEARNING_RATE, fig, ax, artists, color_list):

    print("in seq delta training until converge")

    input_samples  = len(X[0,:])
    convergence_error = CONVERGENCE_FACTOR*LEARNING_RATE

    delta_W = np.zeros((1, IN_DIM+1))

    num_epochs = 0
    is_first_it = True
    while (np.sum(delta_W) > convergence_error) or (is_first_it):
        is_first_it = False

        for i in range(input_samples):

            #Compute gradient
            delta_W = LEARNING_RATE * (T[i] - np.matmul(W.T, X[:,i]))         
        
            #Update weights
            W += delta_W
        


        plot_scatter_line(X, W, fig, ax, artists, color_list)

        print("num_epochs", num_epochs)
        num_epochs += 1


    print("num_epochs", num_epochs)
    return W, num_epochs
        


def batch_delta_training_until_convergence(W, X, T, LEARNING_RATE, fig, ax, artists, color_list):
    input_samples  = len(X[0,:])
    convergence_error = CONVERGENCE_FACTOR*LEARNING_RATE


    delta_W = np.zeros((1, IN_DIM+1))

    num_epochs = 0
    is_first_it = True
    while np.sum(delta_W) > convergence_error or is_first_it:
        is_first_it = False

        #Compute gradient
        delta_W = -LEARNING_RATE * np.matmul((np.matmul(W.T, X) - T), X.T)
    
        #Update weights
        W += delta_W

        plot_scatter_line(X, W, fig, ax, artists, color_list)

        num_epochs += 1
    

    return W, num_epochs


def delta_training(W, X, T, LEARNING_RATE, fig, ax, artists, color_list):
    #Trainig

    input_samples  = len(X[0,:])

    delta_W = np.zeros((1, IN_DIM+1))
    for epoch in range(NUM_EPOCHS):

        for i in range(input_samples):
    

            #Compute gradient
            delta_W = LEARNING_RATE * (T[i] - np.matmul(W.T, X[:,i]))    
        
        
            #Update weights
            W += delta_W
        
        
        plot_scatter_line(X, W, fig, ax, artists, color_list)
        
    return W


def batch_delta_training(W, X, T, LEARNING_RATE, fig, ax, artists, color_list):
    #Trainig
    for i in range(NUM_EPOCHS):
    
        #Compute gradient
        delta_W = -LEARNING_RATE * np.matmul((np.matmul(W.T, X) - T), X.T)
    
        #Update weights
        W += delta_W

    
        plot_scatter_line(X, W, fig, ax, artists, color_list)
        
    return W




def step_func(W, x):
    
    wx = np.matmul(W.T, x)

    if wx > 0:
        return 1
    else:
        return 0


def perceptron_training(W, X, T, LEARNING_RATE, fig, ax, artists, color_list):
    
    input_samples = len(X[0,:])

    delta_W = np.zeros((1, IN_DIM+1))

    for epoch in range(NUM_EPOCHS):

        for i in range(input_samples):

            y_i = step_func(W, X[:,i])

            e = T[i] - y_i

            if e != 0:
                W += (LEARNING_RATE * e * X[:,i])[:, np.newaxis]
                
    
        plot_scatter_line(X, W, fig, ax, artists, color_list)


    return W


def separating_line(W, x0):
    return (W[0]*x0 + W[2]) / W[1]
    
def generate_color_list(T):
    
    color_list = []
    for i in range(len(T)):
        if T[i] == 1:
            color_list.append("red")
        elif T[i] == -1:
            color_list.append("blue")
        else:
            print("Error found in generated Target list")
    

    return color_list


def plot_scatter_line(X, W, fig, ax, artists, color_list):
    
    
    
    granularity = 10
    
    
    #Get max and min X[0] values
    max_x0 = max(X[0])
    min_x0 = min(X[0])
    
    
    x0 = np.linspace(min_x0, max_x0, num=granularity)
    
    ax.scatter(X[0,:], X[1,:], X[2,:], c=color_list)

    container = ax.plot(x0, separating_line(W, x0), c='purple')

    artists.append(container)





