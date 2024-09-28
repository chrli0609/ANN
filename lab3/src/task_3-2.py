import numpy as np
import matplotlib.pyplot as plt
from hopfield import *
from functions import *

HAS_SELF_CONNECTION = True
NUM_NEURONS = 1024
FILEPATH = "../data/pict.dat"
IMG_DIM = 32


model = Hopfield(NUM_NEURONS, HAS_SELF_CONNECTION)

training_data = generate_data(FILEPATH, IMG_DIM, training=True)

visualization_data = generate_data(FILEPATH, IMG_DIM, training=False)

for i in range(len(visualization_data)):
    visualize_img(visualization_data[i], image_num=i)