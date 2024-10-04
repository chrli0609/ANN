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
MAX_ITERATIONS = 400


IS_SYNC = False


model = Hopfield(NUM_NEURONS, HAS_SELF_CONNECTION)

all_patterns = generate_data(FILEPATH, IMG_DIM, is_training=True, num_patterns=TOTAL_NUM_PATTERNS)



print(all_patterns[:NUM_PATTERNS_TO_TRAIN, :])
model.train(all_patterns[:NUM_PATTERNS_TO_TRAIN, :])




print("==============================================================================")
print("######################## Make Sure Stored Patterns are Stable #######################")
print("==============================================================================")


for i in range(NUM_PATTERNS_TO_TRAIN):
    model.recall(all_patterns[i], is_synch=IS_SYNC, max_iterations=MAX_ITERATIONS)




print("==============================================================================")
print("######################## Try Pattern 10 #######################")
print("==============================================================================")


model.yippie = 9
model.recall(all_patterns[9], is_synch=IS_SYNC, max_iterations=MAX_ITERATIONS)



print("==============================================================================")
print("######################## Try Pattern 11 #######################")
print("==============================================================================")



model.yippie = 10
model.recall(all_patterns[10], is_synch=IS_SYNC, max_iterations=MAX_ITERATIONS)

