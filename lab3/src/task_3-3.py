import numpy as np
import matplotlib.pyplot as plt
from hopfield import *
from functions import *

HAS_SELF_CONNECTION = True
NUM_NEURONS = 1024
FILEPATH = "../data/pict.dat"
IMG_DIM = 32
NUM_PATTERNS = 11
SAVE_TO_FILE = False
MAX_ITERATIONS = 10 
NUM_ENERGY_MINIMA = 20


model = Hopfield(NUM_NEURONS, HAS_SELF_CONNECTION)

training_data = generate_data(FILEPATH, IMG_DIM, is_training=True, num_patterns=NUM_PATTERNS)

visualization_data = generate_data(FILEPATH, IMG_DIM, is_training=False, num_patterns=NUM_PATTERNS)

model.train(training_data)

energy_matrix = convert_1024_to_img_dim(model.energy, 32)

smallest_energy_vals = np.argpartition(model.energy, NUM_ENERGY_MINIMA)[:NUM_ENERGY_MINIMA]
np.savetxt("energy_matrix.txt", energy_matrix, fmt="%.6f")

print("smallest_energy_vals", np.sort(model.energy[smallest_energy_vals]))