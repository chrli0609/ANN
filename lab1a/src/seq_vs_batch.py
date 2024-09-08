from function import *

import matplotlib.pyplot as plt


OUT_FOLDER = "../out/seq_vs_batch/"


def main():
    np.random.seed(42)

    mA = [1.0, 0.5]
    mB = [-1.0, 0.0]
    sigmaA = 0.5
    sigmaB = 0.5
    

    init_W, X, T, X_test, T_test, color_list = generate_random_input_and_weights(IN_DIM, NUM_SAMPLES_PER_CLASS, mA, mB, sigmaA, sigmaB)

    perceptron_T = np.where(T<=0, 0, 1)
    perceptron_T_test = np.where(T<=0, 0, 1)


    min_lr = 0.001
    max_lr = 0.01
    lr_list = np.linspace(min_lr, max_lr, 10)

    
    #For sequential
    for lr in lr_list:
        print("in lr:", lr)
        single_layer_perceptron(copy.deepcopy(init_W), X, T, lr, color_list, "delta_training", OUT_FOLDER)


    #For batch
    batch_epochs_to_convergence_list = []
    for lr in lr_list:
        single_layer_perceptron(copy.deepcopy(init_W), X, T, lr, color_list, "batch_delta_training", OUT_FOLDER)


    

main()