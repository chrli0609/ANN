from function import *




OUT_FOLDER = "../out/removed_bias_mA_-1_-0.5_mB_1_0/"


def main():
    np.random.seed(42)
    
    lr = 0.001

    

    
    mA = [1.0, 0.5]
    mB = [-1.0, 0.0]

    mA_no_bias = [0.4, 0]
    mB_no_bias = [-0.4, 0]
    sigmaA = 0.5
    sigmaB = 0.5
    

    init_W, X, T, X_test, T_test, color_list = generate_random_input_and_weights(IN_DIM, NUM_SAMPLES_PER_CLASS, mA_no_bias, mB_no_bias, sigmaA, sigmaB)


    #Print axis limits
    print("Axis limits")
    print("x (", min(X[0])-OFFSET, " - ", max(X[0])+OFFSET,")")
    print("y (", min(X[1])-OFFSET, " - ", max(X[1])+OFFSET,")")

    
    


    print("W before delta", init_W)

    #Remove Bias
    num_W_rows, num_W_cols = init_W.shape
    init_W_no_bias = init_W[:num_W_rows-1, :]

    num_input_rows, num_input_cols = X.shape
    X_no_bias = X[:num_input_rows-1,:]
    
    print("init_W_no_bias", init_W_no_bias)
    print("X_no_bias", X_no_bias)

    #Removed bias
    print("Delta REMOVED bias, LEARNING RATE:", lr)
    single_layer_perceptron(copy.deepcopy(init_W_no_bias), X_no_bias, T, lr, color_list, "batch_delta_training", OUT_FOLDER)

    



    print("init_W with bias", init_W)
    print("X with bias", X)

    #With bias
    print("Delta WITH bias, LEARNING RATE:", lr)
    single_layer_perceptron(copy.deepcopy(init_W), X, T, lr, color_list, "batch_delta_training", OUT_FOLDER)






main()


