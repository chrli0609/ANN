import numpy as np
import matplotlib.pyplot as plt
from hopfield import *
from functions import *

HAS_SELF_CONNECTION = True
NUM_NEURONS = 1024
FILEPATH = "../data/pict.dat"
IMG_DIM = 32
NUM_PATTERNS = 3
SAVE_TO_FILE = False
MAX_ITERATIONS = 10


model = Hopfield(NUM_NEURONS, HAS_SELF_CONNECTION)

training_data = generate_data(FILEPATH, IMG_DIM, is_training=True, num_patterns=NUM_PATTERNS)

visualization_data = generate_data(FILEPATH, IMG_DIM, is_training=False, num_patterns=NUM_PATTERNS)

model.train(training_data)
for i in range(NUM_PATTERNS):
    model.recall(training_data[i], is_synch=False, max_iterations=MAX_ITERATIONS)

"""for i in range(len(visualization_data)):
    visualize_img(visualization_data[i], image_num=i, save_to_file=SAVE_TO_FILE)"""