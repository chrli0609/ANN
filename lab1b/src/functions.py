import numpy as np
import matplotlib.pyplot as plt



IN_DIM = 2
NUM_SAMPLES_PER_CLASS = 100

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



def gen_non_lin_data(in_dim, ndata, mA, mB, sigmaA, sigmaB):
    #ndata = 100

    classA = np.zeros((2, ndata))
    classB = np.zeros((2, ndata))

    classA[0, :] = np.concatenate([
    np.random.randn(round(0.5 * ndata)) * sigmaA - mA[0],
    np.random.randn(round(0.5 * ndata)) * sigmaA + mA[0]
    ])
    classA[1, :] = np.random.randn(ndata) * sigmaA + mA[1]

    classB[0, :] = np.random.randn(ndata) * sigmaB + mB[0]
    classB[1, :] = np.random.randn(ndata) * sigmaB + mB[1]



    
    
    X = np.concatenate((classA, classB), axis=1)
    



    return X


def subsampling_25_from_each_class(all_data):

    num_rows, num_cols = all_data.shape
    size = int(0.75*num_cols/2)

    classA = all_data[:, :int(num_cols/2)]
    classB = all_data[:, int(num_cols/2):]


    np.random.shuffle(np.transpose(classA))
    np.random.shuffle(np.transpose(classB))

    
    train_A = classA[:,:size]
    train_B = classB[:,:size]

    test_A = classA[:,size:]
    test_B = classB[:,size:]

    

    data_train = np.concatenate((train_A, train_B), axis=1)
    data_test = np.concatenate((test_A, test_B), axis=1)

    
    np.random.shuffle(np.transpose(data_train))
    np.random.shuffle(np.transpose(data_test))





    return data_train, data_test



def subsampling_50_from_classA(all_data):

    num_rows, num_cols = all_data.shape
    size = int(0.75*num_cols/2)

    classA = all_data[:, :int(num_cols/2)]
    classB = all_data[:, int(num_cols/2):]

 
    np.random.shuffle(np.transpose(classA))
    
    train_A = classA[:,:size]
    test_A = classA[:,size:]
    

    data_train = np.concatenate((train_A, classB), axis=1)
    data_test = test_A

    
    np.random.shuffle(np.transpose(data_train))
    np.random.shuffle(np.transpose(data_test))



    return data_train, data_test





    

#20% from a subset of classA for which classA(1,:)<0 and 80% from a
#subset of classA for which classA(1,:)>0
def subsampling_point_2_lt_0_and_point_8_gt_0_from_A(all_data):


    num_rows, num_cols = all_data.shape
    size_group1 = int(0.8*num_cols/2)
    size_group2 = int(0.2*num_cols/2)

    classA = all_data[:, :int(num_cols/2)]
    classB = all_data[:, int(num_cols/2):]


    group1_mask = classA[0,:] < 0
    group2_mask = classA[0,:] > 0


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



def generate_random_non_linear_input_and_weights(in_dim, n, mA, mB, sigmaA, sigmaB, subsampling_function):


    X = gen_non_lin_data(in_dim, n, mA, mB, sigmaA, sigmaB)



    #Target matrix
    t1 = np.ones(n)
    t2 =-np.ones(n)
    T = np.concatenate((t1, t2))
    

    
    #Split some parts of the data for testing
    #split_index = int(train_test_dist * (2*n))
    
    
    #Merge both answer and input into the same matrix
    all_data = np.concatenate((X, T[:, None].T), axis=0)


    training_set, test_set = subsampling_function(all_data)
    
    
    #Test input and test target
    X_test = test_set[:in_dim,:]
    T_test = test_set[-1,:]
    
    
    
    ####### Generate the input X and target T to be used during training #######
    X = training_set[:in_dim,:]
    T = training_set[-1, :]
    


    #Generate color list
    color_list = generate_color_list(T)
    
    
    #Add extra row at bottom for bias
    W = np.random.rand(in_dim+1, 1)


    print("W:", W.shape)
    print("X:", X.shape)
    print("T:", T.shape)
    print("X_text:", X_test.shape)
    print("T_test:", T_test.shape)
    

    return W, X, T, X_test, T_test, color_list






def single_to_double_T(T):
    
    multi_T = []
    for target in T:
        expected_out = [None] * 2
        if target == 1:
            expected_out = [1, 0]
        elif target == -1:
            expected_out = [0, 1]
        else:
            print("An error has been found in Target Vector")

        multi_T.append(expected_out)

    multi_T_np = np.array(multi_T)

    return multi_T_np.T




def max_of_col(O):
    _, num_cols = O.shape

    new_O = []
    for i in range(num_cols):
        each_row = [None] * 2
        if O[0][i] > O[1][i]:
             each_row[0] = 1
             each_row[1] = 0
        else:
            each_row[0] = 0
            each_row[1] = 1
        new_O.append(each_row)

    return np.array(new_O).T


def sign(num):
    return -1 if num < 0 else 1

def accuracy_score(O, T):

    correct = 0
    incorrect = 0
    for i in range(len(O)):
        if sign(O[i]) == sign(T[i]):
            correct += 1
        else:
            incorrect += 1
    
    return correct / (correct + incorrect)
            




def plot_data(X, color_list):
	
	
	plt.scatter(X[0,:], X[1,:], c=color_list)
	plt.show()




