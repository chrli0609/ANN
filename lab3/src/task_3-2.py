import numpy as np
import matplotlib.pyplot as plt
from hopfield import *
from functions import *

HAS_SELF_CONNECTION = True
NUM_NEURONS = 1024
FILEPATH = "../data/pict.dat"
IMG_DIM = 32
TOTAL_NUM_PATTERNS = 11
NUM_PATTERNS_TO_TRAIN = 3
SAVE_TO_FILE = False
MAX_ITERATIONS = 15


model = Hopfield(NUM_NEURONS, HAS_SELF_CONNECTION)

all_patterns = generate_data(FILEPATH, IMG_DIM, is_training=True, num_patterns=TOTAL_NUM_PATTERNS)

visualization_data = generate_data(FILEPATH, IMG_DIM, is_training=False, num_patterns=TOTAL_NUM_PATTERNS)


print(all_patterns[:NUM_PATTERNS_TO_TRAIN, :])
model.train(all_patterns[:NUM_PATTERNS_TO_TRAIN, :])




print("==============================================================================")
print("######################## Make Sure Stored Patterns are Stable #######################")
print("==============================================================================")


for i in range(NUM_PATTERNS_TO_TRAIN):
    model.recall(all_patterns[i], is_synch=False, max_iterations=MAX_ITERATIONS)




print("==============================================================================")
print("######################## Try Pattern 10 #######################")
print("==============================================================================")


model.recall(all_patterns[9], is_synch=False, max_iterations=MAX_ITERATIONS)
model.recall(all_patterns[10], is_synch=False, max_iterations=MAX_ITERATIONS)

