import numpy as np
import matplotlib.pyplot as plt
from hopfield import *
from functions import *

HAS_SELF_CONNECTION = False
NUM_NEURONS = 1024
FILEPATH = "../data/pict.dat"
IMG_DIM = 32
NUM_PATTERNS = 4
SAVE_TO_FILE = False


IS_SYNC = False
MAX_ITER = 10
NUM_TRIES = 10**3


model = Hopfield(NUM_NEURONS, HAS_SELF_CONNECTION)

training_data = generate_data(FILEPATH, IMG_DIM, is_training=True, num_patterns=NUM_PATTERNS)

#visualization_data = generate_data(FILEPATH, IMG_DIM, is_training=False, num_patterns=NUM_PATTERNS)

model.train(training_data)

print("training data")
print(training_data.shape)



unique_local_minimas = get_attractors(model, training_data, IS_SYNC, MAX_ITER, NUM_TRIES)

energies_at_local_minima = []
for lm in unique_local_minimas:
    energies_at_local_minima.append(model.compute_energy(lm))



print("local minima energies\n", np.array(energies_at_local_minima))

