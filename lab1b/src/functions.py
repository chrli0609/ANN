import matplotlib.pyplot as plt






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




def plot_data(X, color_list):
	
	
	plt.scatter(X[0,:], X[1,:], c=color_list)
	plt.show()




