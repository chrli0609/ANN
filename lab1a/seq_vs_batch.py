from function import *

import matplotlib.pyplot as plt



def main():
    np.random.seed(42)

    init_W, X, T, X_test, T_test, color_list = generate_random_input_and_weights(IN_DIM, NUM_SAMPLES_PER_CLASS)

    perceptron_T = np.where(T<=0, 0, 1)
    perceptron_T_test = np.where(T<=0, 0, 1)


    min_lr = 0.0001
    max_lr = 0.1
    lr_list = np.linspace(min_lr, max_lr, int(max_lr/min_lr))

    
    #For sequential
    seq_epochs_to_convergence_list = []
    for lr in lr_list:
        print("in lr:", lr)
        seq_epochs_to_convergence_list.append(single_layer_perceptron(copy.deepcopy(init_W), X, T, lr, color_list, "sequential_delta_training_convergence"))


    #For batch
    batch_epochs_to_convergence_list = []
    for lr in lr_list:
        batch_epochs_to_convergence_list.append(single_layer_perceptron(copy.deepcopy(init_W), X, T, lr, color_list, "batch_delta_training_convergence"))

    plt.plot(seq_epochs_to_convergence_list)
    plt.plot(batch_epochs_to_convergence_list)
    plt.show()

    

main()