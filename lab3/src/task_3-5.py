import numpy as np
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt
from hopfield import Hopfield
from functions import *

# Data stuff
FILEPATH = "../data/pict.dat"
IMG_DIM = 32
NUM_TRAINING_PATTERNS = 9

# Model params
HAS_SELF_CONNECTION = False
NUM_NEURONS = 1024
IS_SYNC = True
MAX_ITER = 5

# Outputs
SAVE_TO_FILE = True

USE_PATTERN = False
ADD_BIAS = False

if USE_PATTERN:
    subfolder = "pattern"
else:
    subfolder = "random"

FOLDER_PATH = "../out/task_3-5/png/" + subfolder + "/"

if USE_PATTERN:
    all_data = generate_data(FILEPATH, IMG_DIM, is_training=True, num_patterns=NUM_TRAINING_PATTERNS)
else:
    all_data = np.random.choice([-1, 1], size=(NUM_TRAINING_PATTERNS, NUM_NEURONS))

if ADD_BIAS:
    all_data = all_data + np.sign(0.5 + np.random.randn(300, 100))

all_models_list = []

# Only one percentage defined in the list
percentage = 0.3  # Noise level

# Train for each number of training patterns
for i in range(1, NUM_TRAINING_PATTERNS):
    training_data = all_data[0:i, :]
    model = Hopfield(NUM_NEURONS, HAS_SELF_CONNECTION)
    model.train(training_data)

    reconstruct_hamming_sum = 0

    # Create noisy version of training data
    noisy_training_data = scramble_data(training_data, percentage)

    for k in range(len(noisy_training_data)):
        # Recall the distorted data
        model.recall(noisy_training_data[k], IS_SYNC, MAX_ITER)
        reconstruct_hamming_sum += hamming(training_data[k], model.neurons)

        if USE_PATTERN:
            # Visualize noisy data
            img = convert_1024_to_img_dim(noisy_training_data[k], img_dim=32)
            visualize_img_path(img, FOLDER_PATH + "/p1-" + str(i+1) + "/", "noisy_p"+str(k+1)+"_"+str(percentage), SAVE_TO_FILE, "p1-" + str(i+1) + " noisy p"+str(k+1)+"_"+str(percentage))

            # Visualize reconstruction
            img = convert_1024_to_img_dim(model.neurons, img_dim=32)
            visualize_img_path(img, FOLDER_PATH + "/p1-" + str(i+1) + "/", "reconstructed_p"+str(k+1)+"_"+str(percentage), SAVE_TO_FILE, "p1-" + str(i+1) + " reconstructed p"+str(k+1)+"_"+str(percentage))

    avg_hamming_distance = reconstruct_hamming_sum / len(noisy_training_data)
    all_models_list.append(1024*np.round(avg_hamming_distance, 4))

# Ensure any previous plots are cleared
plt.clf()

# Create a new figure to ensure separation from image plots
plt.figure()

# Plot Hamming distance for the single noise level
plt.plot(range(1, NUM_TRAINING_PATTERNS), all_models_list, label=f"{int(percentage*100)}% Noise")
plt.title("Hamming Distance vs Number of Patterns Stored")
plt.xlabel("Number of Patterns")
plt.ylabel("Hamming Distance")
plt.legend()
plt.show()

print(all_models_list)
