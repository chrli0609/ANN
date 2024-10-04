import numpy as np
from scipy.spatial.distance import hamming




from hopfield import Hopfield
from functions import *


#Data stuff
FILEPATH = "../data/pict.dat"
IMG_DIM = 32
NUM_TRAINING_PATTERNS = 300

#Model params
HAS_SELF_CONNECTION = False
NUM_NEURONS = 100
IS_SYNC = True
MAX_ITER = 5


#Outputs
SAVE_TO_FILE = True



###############################################################
###############################################################
###############################################################
###############################################################
USE_PATTERN = False                      # <---------------- LOOK HEREEREREERERERR
ADD_BIAS = True

###############################################################
###############################################################
###############################################################
###############################################################


if USE_PATTERN:
    subfolder = "pattern"
else:
    subfolder = "random300"

FOLDER_PATH = "../out/task_3-5/png/"+subfolder+"/"




if USE_PATTERN:
    all_data = generate_data(FILEPATH, IMG_DIM, is_training=True, num_patterns=NUM_TRAINING_PATTERNS)
else:
    all_data = np.random.choice([-1, 1], size=(NUM_TRAINING_PATTERNS, NUM_NEURONS))

if ADD_BIAS:
    all_data = all_data + np.sign(0.5 + np.random.randn(300, 100))




all_models_list = []
model_name_list = []


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



    per_noise_lvl = []

    # Generate each different level of noise
    for j in range(len(percentage_list)):
        

        #Get the noisy version of the training data
        percentage = np.round(percentage_list[j], 2)

        noisy_training_data = scramble_data(training_data, percentage)




        #Try to reconstruct each level of noise for every pattern
        reconstruct_hamming_sum = 0
        for k in range(len(noisy_training_data)):
            #Do recall with distorted data
            model.recall(noisy_training_data[k], IS_SYNC, MAX_ITER)

            #print("hamming(training_data["+str(k)+"], model.neurons)", hamming(training_data[k], model.neurons))

            reconstruct_hamming_sum += hamming(training_data[k], model.neurons)

            if USE_PATTERN:
                #Visualize distorted data
                img = convert_1024_to_img_dim(noisy_training_data[k], img_dim=32)
                visualize_img_path(img, FOLDER_PATH + "/p1-" + str(i+1) + "/", "noisy_p"+str(k+1)+"_"+str(percentage), SAVE_TO_FILE, "p1-" + str(i+1) + " noisy p"+str(k+1)+"_"+str(percentage))


                #Visualize reconstruction
                img = convert_1024_to_img_dim(model.neurons, img_dim=32)
                visualize_img_path(img, FOLDER_PATH + "/p1-" + str(i+1) + "/", "reconstructed_p"+str(k+1)+"_"+str(percentage), SAVE_TO_FILE, "p1-" + str(i+1) + " reconstructed p"+str(k+1)+"_"+str(percentage))



        
        per_noise_lvl.append(reconstruct_hamming_sum)


    all_models_list.append(per_noise_lvl)




print(all_models_list)
print()
print()

from prettytable import PrettyTable
t = PrettyTable(['Num Mem', '10%', '20%', '30%', '40%'])
for i in range(len(all_models_list)):

    all_models_list[i].insert(0, i+3)
    
    t.add_row(all_models_list[i])
print(t)

