import numpy as np
from scipy.spatial.distance import hamming


from hopfield import Hopfield
from functions import *


#Data stuff
FILEPATH = "../data/pict.dat"
IMG_DIM = 32
NUM_TRAINING_PATTERNS = 9

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

USE_PATTERN = False







if USE_PATTERN:
    subfolder = "pattern"
else:
    subfolder = "random"

FOLDER_PATH = "../out/task_3-5/png/"+subfolder+"/"




if USE_PATTERN:
    all_data = generate_data(FILEPATH, IMG_DIM, is_training=True, num_patterns=NUM_TRAINING_PATTERNS)
else:
    all_data = np.random.choice([-1, 1], size=(NUM_TRAINING_PATTERNS, NUM_NEURONS))



all_models_list = []


percentage_list = [0.1, 0.2, 0.3, 0.4]
#percentage_list = [0.3, 0.4]

#Inrease the number of memories
for i in range(2, NUM_TRAINING_PATTERNS):

    
    #Define the set of patterns we want our network to train
    training_data = all_data[0:i,:]


    #Generate Pattern
    model = Hopfield(NUM_NEURONS, HAS_SELF_CONNECTION)


    #Train our model based on the patterns
    model.train(training_data)



    # Generate each different level of noise
    for j in range(len(percentage_list)):
        

        #Get the noisy version of the training data
        percentage = np.round(percentage_list[j], 2)

        noisy_training_data = scramble_data(training_data, percentage, NOISE_MEAN, NOISE_STD)




        #Try to reconstruct each level of noise for every pattern
        per_noise_lvl_list = []
        for k in range(len(noisy_training_data)):
            #Do recall with distorted data
            model.recall(noisy_training_data[k], IS_SYNC, MAX_ITER)


            per_noise_lvl_list.append(hamming(noisy_training_data[k], model.neurons))
            if USE_PATTERN:
                #Visualize distorted data
                img = convert_1024_to_img_dim(noisy_training_data[k], img_dim=32)
                visualize_img_path(img, FOLDER_PATH + "/p1-" + str(i+1) + "/", "noisy_p"+str(k+1)+"_"+str(percentage), SAVE_TO_FILE, "p1-" + str(i+1) + " noisy p"+str(k+1)+"_"+str(percentage))


                #Visualize reconstruction
                img = convert_1024_to_img_dim(model.neurons, img_dim=32)
                visualize_img_path(img, FOLDER_PATH + "/p1-" + str(i+1) + "/", "reconstructed_p"+str(k+1)+"_"+str(percentage), SAVE_TO_FILE, "p1-" + str(i+1) + " reconstructed p"+str(k+1)+"_"+str(percentage))



        
        all_models_list.append(per_noise_lvl_list)







from prettytable import PrettyTable
t = PrettyTable(['Name', '10%', '20%', '30%', '40%'])
for i in range(len(all_models_list)):
    t.add_rows(all_models_list[i])
print(t)

