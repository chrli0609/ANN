from function import *
import numpy as np



#OUT_FOLDER = "./non_lin_batch_25_from_each_class/"
#OUT_FOLDER = "./non_lin_batch_50_from_A/"
#OUT_FOLDER = "./non_lin_batch_50_from_B/"
OUT_FOLDER = "./non_lin_batch_point_2_lt_0_and_point_8_gt_0_from_A/"


def main():
    np.random.seed(42)
    

    min_lr = 0.0001
    max_lr = 0.1
    #lr_list = np.linspace(min_lr, max_lr, int(max_lr/min_lr))
    lr_list = lr_list_log(min_lr, max_lr)
    

    ## Format data
    mA = np.array([1.0, 0.3])
    sigmaA = 0.2
    mB = np.array([0.0, -0.1])
    sigmaB = 0.3
    
    
    #subsampling_25_from_each_class
    #subsampling_50_from_classA
    #subsampling_50_from_classB
    #subsampling_point_2_lt_0_and_point_8_gt_0_from_A

    init_W, X, T, X_test, T_test, color_list = generate_random_non_linear_input_and_weights(IN_DIM, NUM_SAMPLES_PER_CLASS, mA, mB, sigmaA, sigmaB, "subsampling_point_2_lt_0_and_point_8_gt_0_from_A")



    perceptron_T = np.where(T==-1, 0, 1)
    perceptron_T_test = np.where(T==-1, 0, 1)


    #Print axis limits
    print("Axis limits")
    print("x (", min(X[0])-OFFSET, " - ", max(X[0])+OFFSET,")")
    print("y (", min(X[1])-OFFSET, " - ", max(X[1])+OFFSET,")")


    print("W before delta", init_W)
    #BATCH MODE delta rule
    for lr in lr_list:
        print("Delta Rule Training with LEARNING RATE:", lr)
        single_layer_perceptron(copy.deepcopy(init_W), X, T, lr, color_list, "batch_delta_training", OUT_FOLDER)

        
    print("W before perceptron", init_W)

    #Perceptron learning
    for lr in lr_list:
        print("Perceptron Learning with LEARNING RATE:", lr)
        single_layer_perceptron(copy.deepcopy(init_W), X, perceptron_T, lr, color_list, "perceptron_learning", OUT_FOLDER)



main()
