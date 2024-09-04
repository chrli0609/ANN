from function import *

import matplotlib.pyplot as plt


OUT_FOLDER = "./seq_vs_batch/"


def main():
    np.random.seed(42)

    mA = [1.0, 0.5]
    mB = [-1.0, 0.0]
    sigmaA = 0.5
    sigmaB = 0.5
    

    init_W, X, T, X_test, T_test, color_list = generate_random_input_and_weights(IN_DIM, NUM_SAMPLES_PER_CLASS, mA, mB, sigmaA, sigmaB)

    perceptron_T = np.where(T<=0, 0, 1)
    perceptron_T_test = np.where(T<=0, 0, 1)


    min_lr = 0.0001
    max_lr = 0.63
    lr_list = np.linspace(min_lr, max_lr, 50)

    print("lr_list", lr_list)
    
    #For sequential
    seq_epochs_to_convergence_list = []
    for lr in lr_list:
        print("in lr:", lr)
        seq_epochs_to_convergence_list.append(single_layer_perceptron(copy.deepcopy(init_W), X, T, lr, color_list, "sequential_delta_training_convergence", OUT_FOLDER))



    plt.plot(lr_list, seq_epochs_to_convergence_list)
    plt.title("Epochs until convergence (error < " + str(CONVERGENCE_FACTOR) + ")")
    #plt.plot(lr_list, batch_epochs_to_convergence_list)
    plt.xlabel ('Learning Rate')
    plt.ylabel ('Number of Epochs until convergence')

    plt.show()




    min_lr_batch = 0.00001
    max_lr_batch = 0.0001
    lr_list_batch = np.linspace(min_lr_batch, max_lr_batch, 10)

    print("lr_list_batch", lr_list_batch)

    #For batch
    batch_epochs_to_convergence_list = []
    for lr_batch in lr_list_batch:
        batch_epochs_to_convergence_list.append(single_layer_perceptron(copy.deepcopy(init_W), X, T, lr_batch, color_list, "batch_delta_training_convergence", OUT_FOLDER))




    plt.plot(lr_list_batch, batch_epochs_to_convergence_list)
    plt.title("Epochs until convergence (error < " + str(CONVERGENCE_FACTOR) + ")")
    plt.xlabel ('Learning Rate')
    plt.ylabel ('Number of Epochs until convergence')

    plt.show()
    

    

main()



