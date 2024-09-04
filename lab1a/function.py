import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

import sys
import copy
import math


OFFSET = 0.2

#is_delta_rule = False

train_test_dist = 0.8
NUM_EPOCHS = 20
IN_DIM = 2

#Num samples per class
NUM_SAMPLES_PER_CLASS = 100
CONVERGENCE_FACTOR = 0.0005




def lr_list_log(min_lr, max_lr):

    lr_samples = int(math.log10(int(max_lr/min_lr)))

    #Generate list of learning rates to test
    lr_list = [min_lr]
    for i in range(lr_samples):
        curr_value = lr_list[-1]
        lr_list.append(curr_value*10)

    print("lr_list", lr_list)
    return lr_list


def gen_non_lin_data(in_dim, n, mA, mB, sigmaA, sigmaB):
    ndata = 100

    classA = np.zeros((2, ndata))
    classB = np.zeros((2, ndata))

    classA[0, :] = np.concatenate([
    np.random.randn(round(0.5 * ndata)) * sigmaA - mA[0],
    np.random.randn(round(0.5 * ndata)) * sigmaA + mA[0]
    ])
    classA[1, :] = np.random.randn(ndata) * sigmaA + mA[1]

    classB[0, :] = np.random.randn(ndata) * sigmaB + mB[0]
    classB[1, :] = np.random.randn(ndata) * sigmaB + mB[1]




    #Add an extra row in bottom of ones to handle bias
    bias_row = np.ones((1, 2*n))
    
    
    data = np.concatenate((classA, classB), axis=1)
    

    X = np.concatenate((data, bias_row), axis=0)



    return X


def subsampling_25_from_each_class(all_data):
    ndata = 100
    num_rows, num_cols = all_data.shape
    size = int(0.75*ndata)

    classA = all_data[:, :int(num_cols/2)]
    classB = all_data[:, int(num_cols/2):]

    print("classA", classA)
    print("classB", classB)

    np.random.shuffle(np.transpose(classA))
    np.random.shuffle(np.transpose(classB))

    
    train_A = classA[:,:size]
    train_B = classB[:,:size]

    test_A = classA[:,size:]
    test_B = classB[:,size:]

    #sub_A = np.random.choice(classA, size, replace=False)
    #sub_B = np.random.choice(classB, size, replace=False)

    print("A",train_A.shape)
    print("B",train_B.shape)

    

    data_train = np.concatenate((train_A, train_B), axis=1)
    data_test = np.concatenate((test_A, test_B), axis=1)

    
    np.random.shuffle(np.transpose(data_train))
    np.random.shuffle(np.transpose(data_test))



    print("data_train", data_train)
    print("data_test", data_test)

    return data_train, data_test



def subsampling_50_from_classA(all_data):
    ndata = 100
    num_rows, num_cols = all_data.shape
    size = int(0.75*ndata)

    classA = all_data[:, :int(num_cols/2)]
    classB = all_data[:, int(num_cols/2):]

    print("classA", classA)
    print("classB", classB)

    np.random.shuffle(np.transpose(classA))
    
    train_A = classA[:,:size]
    test_A = classA[:,size:]

    #sub_A = np.random.choice(classA, size, replace=False)
    #sub_B = np.random.choice(classB, size, replace=False)

    

    data_train = np.concatenate((train_A, classB), axis=1)
    data_test = test_A

    
    np.random.shuffle(np.transpose(data_train))
    np.random.shuffle(np.transpose(data_test))



    return data_train, data_test




def subsampling_50_from_classB(all_data):
    ndata = 100
    num_rows, num_cols = all_data.shape
    size = int(0.75*ndata)

    classA = all_data[:, :int(num_cols/2)]
    classB = all_data[:, int(num_cols/2):]


    np.random.shuffle(np.transpose(classB))
    
    train_B = classB[:,:size]
    test_B = classB[:,size:]

    #sub_A = np.random.choice(classA, size, replace=False)
    #sub_B = np.random.choice(classB, size, replace=False)

    

    data_train = np.concatenate((classA, train_B), axis=1)
    data_test = test_B

    
    np.random.shuffle(np.transpose(data_train))
    np.random.shuffle(np.transpose(data_test))



    return data_train, data_test

    

#20% from a subset of classA for which classA(1,:)<0 and 80% from a
#subset of classA for which classA(1,:)>0
def subsampling_point_2_lt_0_and_point_8_gt_0_from_A(all_data):


    ndata = 100
    num_rows, num_cols = all_data.shape
    size_group1 = int(0.8*ndata)
    size_group2 = int(0.2*ndata)

    classA = all_data[:, :int(num_cols/2)]
    classB = all_data[:, int(num_cols/2):]



    print("classA", classA)

    group1_mask = classA[0,:] < 0
    group2_mask = classA[0,:] > 0

    print("group1_mask", group1_mask)
    print("group2_mask", group2_mask)

    group1_list = classA[:,group1_mask]
    group2_list = classB[:,group2_mask]

    #Randomize within group 1 and group 2
    np.random.shuffle(np.transpose(group1_list))
    np.random.shuffle(np.transpose(group2_list))


    #Take 80% of group1 for training
    train_group1 = group1_list[:,:size_group1]
    #Take 20% of group1 for testing
    test_group1 = group1_list[:,size_group1:]

    #Take 20% of group2 for training
    train_group2 = group2_list[:,:size_group2]
    #Take 80% of group2 for testing
    test_group2 = group2_list[:, size_group2:]
    
    

    data_train = np.concatenate((train_group1, train_group2, classB), axis=1)
    data_test = np.concatenate((test_group1, test_group2), axis=1)

    
    np.random.shuffle(np.transpose(data_train))
    np.random.shuffle(np.transpose(data_test))



    return data_train, data_test



def generate_random_non_linear_input_and_weights(in_dim, n, mA, mB, sigmaA, sigmaB, subsampling_type):


    X = gen_non_lin_data(in_dim, n, mA, mB, sigmaA, sigmaB)




    #Target matrix
    t1 = np.ones(n)
    t2 =-np.ones(n)
    T = np.concatenate((t1, t2))
    
    
    
    
    #Split some parts of the data for testing
    #split_index = int(train_test_dist * (2*n))
    
    
    #Merge both answer and input into the same matrix
    all_data = np.concatenate((X, T[:, None].T), axis=0)


    if subsampling_type == "subsampling_25_from_each_class":
        training_set, test_set = subsampling_25_from_each_class(all_data)
    elif subsampling_type == "subsampling_50_from_classA":
        training_set, test_set = subsampling_50_from_classA(all_data)
    elif subsampling_type == "subsampling_50_from_classB":
        training_set, test_set = subsampling_50_from_classB(all_data)
    elif subsampling_type == "subsampling_point_2_lt_0_and_point_8_gt_0_from_A":
        training_set, test_set = subsampling_point_2_lt_0_and_point_8_gt_0_from_A(all_data)
    else:
        print("ERRORRRR: Incorrect subsampling type string")
    
    
    #Shuffle the data
    #np.random.shuffle(np.transpose(all_data))
    
    #print("split_index", split_index)
    
    #Split into training and testing set
    #training_set, test_set = all_data[:, :split_index], all_data[:,split_index:]
    
    
    #print("all_data[:,:split_index]", all_data[:,:split_index].shape)
    #print("all_data[:,split_index:]", all_data[:,split_index:].shape)
    
    print("training set", training_set)
    
    
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

    


def generate_random_input_and_weights(in_dim, n, mA, mB, sigmaA, sigmaB):
    

    #mA = [1.0, 0.5]
    #mB = [-1.0, 0.0]
    
    #sigmaA = 0.5
    #sigmaB = 0.5
    
    
    
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


def single_layer_perceptron(W, X, T, LEARNING_RATE, color_list, training_type, OUT_FOLDER):

    if training_type == "delta_training":
        plot_title = "Delta Rule Single Layer Perceptron with Learning Rate "
        filepath = "SLP_Delta_rule_lr_"
    elif training_type == "batch_delta_training":
        if len(X) == 3:
            plot_title = "Batch Delta Rule Single Layer Perceptron with Learning Rate "
            filepath = "SLP_Batch_Delta_rule_lr_"
        elif len(X) == 2:
            plot_title = "Batch Delta Rule NO BIAS Single Layer Perceptron with Learning Rate "
            filepath = "SLP_Batch_Delta_rule_NO_BIAS_lr_"
        else:
            print("ERROR INCORRECT INPUT DIMENSIOONS")

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
    if "convergence" not in training_type:
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot()
        ax.set(xlim=(min(X[0])-OFFSET, max(X[0])+OFFSET), ylim=(min(X[1])-OFFSET, max(X[1])+OFFSET))
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        plt.title(plot_title_str)
        ax.grid(True, which='both')
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        artists = []
    

    
    if training_type == "delta_training":
        W = delta_training(W, X, T, LEARNING_RATE, fig, ax, artists, color_list)
    elif training_type == "batch_delta_training":
        W = batch_delta_training(W, X, T, LEARNING_RATE, fig, ax, artists, color_list)
    elif training_type == "perceptron_learning":
        W = perceptron_training(W, X, T, LEARNING_RATE, fig, ax, artists, color_list)
    elif training_type == "sequential_delta_training_convergence":
        W, num_epochs = sequential_delta_training_until_converge(W, X, T, LEARNING_RATE, color_list)
        #print("Artists seq", artists)
    elif training_type == "batch_delta_training_convergence":
        W, num_epochs = batch_delta_training_until_convergence(W, X, T, LEARNING_RATE, color_list)
        #print("Artists batch", artists)
    else:
        print("Error: Training Rule Not Found")
    
    
    
    
    
    print("artists", len(artists))
    
    
    #Final Weights
    print("Final Weights:\n", W)

    if "convergence" not in training_type:
        ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=800)
        plt.show(block=False)
    

        f = OUT_FOLDER + filepath_str + '.gif'
        writergif = animation.PillowWriter(fps=5)
        ani.save(f, writer=writergif)
    else:
        return num_epochs


def elementwise_and(list_of_bools):

    for boolean in list_of_bools:
        if boolean == False:
            return False
    
    return True


def sequential_delta_training_until_converge(W, X, T, LEARNING_RATE, color_list):

    print("in seq delta training until converge")

    input_samples  = len(X[0,:])
    convergence_error = CONVERGENCE_FACTOR

    delta_W = np.zeros((1, IN_DIM+1))

    num_epochs = 0
    is_first_it = True

    sum_delta = sys.maxsize

    while True:

        if (np.where(abs(sum_delta) < convergence_error, True, False).all()) and not is_first_it:
            break


        is_first_it = False

        print("==========================NEW EPOCH========================", num_epochs)

        sum_delta = 0
        for i in range(input_samples):

            #Compute gradient
            delta_W = (LEARNING_RATE * (T[i] - np.matmul(W.T, X[:,i])) * X[:,i])[:, np.newaxis]

            sum_delta += delta_W
        
            #Update weights
            W += delta_W



        print("sum_delta", sum_delta)
        print("W", W)
        print("convergence_error", convergence_error)

        print("np.where(abs(sum_delta) > convergence_error, True, False).all()", np.where(abs(sum_delta) > convergence_error, True, False).all())
        

        #plot_scatter_line(X, W, fig, ax, artists, color_list)

        num_epochs += 1

        print("num_epochs", num_epochs)



    return W, num_epochs
        


def batch_delta_training_until_convergence(W, X, T, LEARNING_RATE, color_list):
    input_samples  = len(X[0,:])
    convergence_error = CONVERGENCE_FACTOR

    delta_W = np.zeros((1, IN_DIM+1))

    num_epochs = 0
    is_first_it = True

    while True:

        if (np.where(abs(delta_W) < convergence_error, True, False).all()) and not is_first_it:
            break

        is_first_it = False

        print("W", W)
        if (math.isnan(W[0])):
            print("rgagowhaoihergaoeraeorgaeorgaeorgaehrgioaeghaeiohgoaehgoh")
            print("Num epoch", num_epochs)
            print("Learning rate", LEARNING_RATE)
            quit()
        #Compute gradient
        delta_W = -LEARNING_RATE * np.matmul((np.matmul(W.T, X) - T), X.T)


        #Update weights
        W += delta_W.T

        #plot_scatter_line(X, W, fig, ax, artists, color_list)

        num_epochs += 1
    

    return W, num_epochs


def delta_training(W, X, T, LEARNING_RATE, fig, ax, artists, color_list):
    #Trainig

    input_samples  = len(X[0,:])

    delta_W = np.zeros((IN_DIM+1, 1))
    for epoch in range(NUM_EPOCHS):

        for i in range(input_samples):
    

            #Compute gradient
            delta_W = (LEARNING_RATE * (T[i] - np.matmul(W.T, X[:,i])) * X[:,i])[:, np.newaxis]
        

            print("delta_W", delta_W)
            print("W", W)

            #Update weights
            W += delta_W
        
        
        plot_scatter_line(X, W, fig, ax, artists, color_list)
        
    return W


def batch_delta_training(W, X, T, LEARNING_RATE, fig, ax, artists, color_list):
    #Trainig

    delta_W = np.zeros((IN_DIM+1, 1))

    for i in range(NUM_EPOCHS):
    
        #Compute gradient
        delta_W = -LEARNING_RATE * np.matmul((np.matmul(W.T, X) - T), X.T)
    
        #Update weights
        W += delta_W.T

    
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
    correctly_classified_A = 0
    incorrectly_classified_A = 0
    correctly_classified_B = 0
    incorrectly_classified_B = 0

    for epoch in range(NUM_EPOCHS):

        for i in range(input_samples):

            #print("epoch num:", epoch, "sample num:", i)

            y_i = step_func(W, X[:,i])

            e = T[i] - y_i

            #print("e", e)

            if e != 0:
                W += (LEARNING_RATE * e * X[:,i])[:, np.newaxis]

        plot_scatter_line(X, W, fig, ax, artists, color_list)

    for i in range(input_samples):
        y_i = step_func(W, X[:,i])

        
        e = T[i] - y_i
        if (y_i == 1):
            if (e != 0):
                incorrectly_classified_A += 1
            else:
                correctly_classified_A += 1
        else:
            if (e != 0):
                incorrectly_classified_B += 1
            else:
                correctly_classified_B += 1

    print("correctly classified", correctly_classified_B)
    print("incorrectly classified", incorrectly_classified_B)
    print("===================================================")
    print("Correctly classified class A: ", float(correctly_classified_A/(incorrectly_classified_A+correctly_classified_A)))
    if correctly_classified_B == 0 and incorrectly_classified_B == 0:
        print("srjg")
    else:
        print("Correctly classified class B: ", float(correctly_classified_B/(incorrectly_classified_B+correctly_classified_B)))
    print("===================================================")

    return W


def separating_line(W, x0):
    if len(W) == 3:
        return -(W[0]*x0 + W[2]) / W[1]
    elif len(W) == 2:
        return -(W[0]*x0) / W[1]
    else:
        print("Error: undefined input dimensions found")


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
    
    if len(X) == 3:
        ax.scatter(X[0,:], X[1,:], X[2,:], c=color_list)
    elif len(X) == 2:
        ax.scatter(X[0,:], X[1,:], c=color_list)
    else:
        print("Error: undefined input dimensions found")

    container = ax.plot(x0, separating_line(W, x0), c='purple')


    artists.append(container)





