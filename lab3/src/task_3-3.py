import numpy as np
import matplotlib.pyplot as plt
from hopfield import *
from functions import *

HAS_SELF_CONNECTION = False
NUM_NEURONS = 1024
FILEPATH = "../data/pict.dat"
IMG_DIM = 32
NUM_PATTERNS = 11
NUM_TRAIN_PATTERNS = 3
SAVE_TO_FILE = False


IS_SYNC = True
MAX_ITER = 5
NUM_TRIES = 10**2



training_data = generate_data(FILEPATH, IMG_DIM, is_training=True, num_patterns=NUM_PATTERNS)

#visualization_data = generate_data(FILEPATH, IMG_DIM, is_training=False, num_patterns=NUM_PATTERNS)

#training_data = np.vstack((training_data[:3, :], training_data[4:, :]))
training_data = training_data[0:NUM_TRAIN_PATTERNS,:]



model = Hopfield(NUM_NEURONS, HAS_SELF_CONNECTION)
model.train(training_data)

print("training data")
print(training_data.shape)



unique_local_minimas = get_attractors(model, training_data, IS_SYNC, MAX_ITER, NUM_TRIES)

energies_at_local_minima = []
i=0
for lm in unique_local_minimas:
    energies_at_local_minima.append(model.compute_energy(lm))
    img = convert_1024_to_img_dim(lm, 32)
    visualize_img(img, 2, i, True)
    #print()
    i+=1



print("local minima energies\n", np.array(energies_at_local_minima))





print("==============================================================================")
print("######################## Random weights #######################")
print("==============================================================================")

random_weight_model = Hopfield(NUM_NEURONS, HAS_SELF_CONNECTION)


random_weight_model.recall(np.random.choice([-1, 1], size=NUM_NEURONS), IS_SYNC, MAX_ITER)
final_state_random = random_weight_model.neurons

rand_img = convert_1024_to_img_dim(final_state_random, 32)
visualize_img(rand_img, 3, 10, True)



print("==============================================================================")
print("######################## Random weights Symmetric #######################")
print("==============================================================================")


random_weight_symmetric_model = Hopfield(NUM_NEURONS, HAS_SELF_CONNECTION)
random_weight_symmetric_model.weights = 0.5 * (random_weight_symmetric_model.weights + random_weight_symmetric_model.weights.T)


random_weight_symmetric_model.recall(np.random.choice([-1, 1], size=NUM_NEURONS), IS_SYNC, MAX_ITER)


final_state_random_symmetric_ = random_weight_symmetric_model.neurons


rand_symmetric_img = convert_1024_to_img_dim(final_state_random_symmetric_, 32)
visualize_img(rand_symmetric_img, 4, 10, True)
