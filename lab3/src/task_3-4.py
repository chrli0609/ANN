import numpy as np


from hopfield import Hopfield
from functions import *



#Data stuff
FILEPATH = "../data/pict.dat"
IMG_DIM = 32

NUM_PATTERNS = 11
NUM_TRAIN_PATTERNS = 3



#Model params
HAS_SELF_CONNECTION = False
NUM_NEURONS = 1024
IS_SYNC = True
MAX_ITER = 5


#Noise params
NOISE_MEAN = 0
NOISE_STD = 1



#Outputs
SAVE_TO_FILE = True
FOLDER_PATH = "../out/task_3-4/png/"





all_data = generate_data(FILEPATH, IMG_DIM, is_training=True, num_patterns=NUM_PATTERNS)
training_data = all_data[0:NUM_TRAIN_PATTERNS,:]



percentage_list = np.arange(0, 1, 0.1)
for j in range(len(percentage_list)):
    percentage = np.round(percentage_list[j], 2)
    noisy_training_data = scramble_data(training_data, percentage, NOISE_MEAN, NOISE_STD)

    
    


    model = Hopfield(NUM_NEURONS, HAS_SELF_CONNECTION)
    model.train(training_data)


    # recall for each pattern
    for i in range(NUM_TRAIN_PATTERNS):
        
        #Do recall with distorted data
        model.recall(noisy_training_data[i], IS_SYNC, MAX_ITER)
        



        #Visualize distorted data
        img = convert_1024_to_img_dim(noisy_training_data[i], img_dim=32)
        visualize_img_path(img, FOLDER_PATH + "/p" + str(i+1) + "/", "noisy_"+str(percentage), SAVE_TO_FILE, "Distorted with "+str(percentage))


        #Visualize reconstruction
        img = convert_1024_to_img_dim(model.neurons, img_dim=32)
        visualize_img_path(img, FOLDER_PATH + "/p" + str(i+1) + "/", "reconstructed_"+str(percentage), SAVE_TO_FILE, "Reconstructed from "+str(percentage) + " noise")









