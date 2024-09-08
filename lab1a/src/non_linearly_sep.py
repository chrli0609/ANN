from function import *
import numpy as np


OUT_FOLDER = "../out/non_linearly_separable/"


def main():
    np.random.seed(42)
    

    min_lr = 0.00001
    max_lr = 0.1
    #lr_list = np.linspace(min_lr, max_lr, int(max_lr/min_lr))
    lr_list = lr_list_log(min_lr, max_lr)
    

    mA = [0.1, 0.3]
    mB = [-0.1, 0.0]
    sigmaA = 0.5
    sigmaB = 0.5
    
    init_W, X, T, X_test, T_test, color_list = generate_random_input_and_weights(IN_DIM, NUM_SAMPLES_PER_CLASS, mA, mB, sigmaA, sigmaB)

    perceptron_T = np.where(T==-1, 0, 1)
    perceptron_T_test = np.where(T==-1, 0, 1)


    #Print axis limits
    print("Axis limits")
    print("x (", min(X[0])-OFFSET, " - ", max(X[0])+OFFSET,")")
    print("y (", min(X[1])-OFFSET, " - ", max(X[1])+OFFSET,")")


    print("W before delta", init_W)
    #Delta rule
    for lr in lr_list:
        print("Delta Rule Training with LEARNING RATE:", lr)
        single_layer_perceptron(copy.deepcopy(init_W), X, T, lr, color_list, "delta_training", OUT_FOLDER)

        
    print("W before perceptron", init_W)

    #Perceptron learning
    for lr in lr_list:
        print("Perceptron Learning with LEARNING RATE:", lr)
        single_layer_perceptron(copy.deepcopy(init_W), X, perceptron_T, lr, color_list, "perceptron_learning", OUT_FOLDER)



main()
