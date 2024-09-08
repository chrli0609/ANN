from functions import *


mA = [0.1, 0.3]
mB = [-0.1, 0.0]
sigmaA = 0.5
sigmaB = 0.5
    
init_W, X, T, X_test, T_test, color_list = generate_random_input_and_weights(IN_DIM, NUM_SAMPLES_PER_CLASS, mA, mB, sigmaA, sigmaB)


plot_data(X, color_list)

