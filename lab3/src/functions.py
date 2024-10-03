import numpy as np
import matplotlib.pyplot as plt

from hopfield import *


import __main__
import os



def generate_data(filepath, img_dim, is_training, num_patterns):
    file = open(filepath, "r")

    pattern_list = file.readline().split(",")
    pattern_vector = np.zeros((num_patterns, 1024))

    counter = 0
    for i in range(num_patterns):
        for j in range(1024):
            pattern_vector[i][j] = int(pattern_list[counter])
            counter += 1
        

    file.close()

    #want 11 matrices that are 32x32
    img_array = np.zeros((num_patterns, img_dim, img_dim))
    for i in range(len(pattern_vector)):
        img_array[i] = convert_1024_to_img_dim(pattern_vector[i], img_dim)

    if (is_training):
        return pattern_vector
    else:
        return img_array


def convert_1024_to_img_dim(vector, img_dim):
    img = np.zeros((img_dim, img_dim))
    for i in range(img_dim):

        for j in range(img_dim):
            img[i,j] = vector[i*img_dim+j]

    return img

def visualize_img(image, image_num, pattern_num, save_to_file):
    #alternative approach
    image = image*255

    plt.imshow(image)

    plt.gray()

    taskname = os.path.basename(__main__.__file__).strip(".py")

    if save_to_file:
        path = "../out/" + taskname + "/png/p" + str(pattern_num) + "/"
        if not os.path.isdir(path):
            os.makedirs(path)
        plt.savefig(path+str(image_num)+".png")
    else:
        plt.show()



def get_attractors(model, train_data, IS_SYNC, MAX_ITER, NUM_TRIES):
    
    random_inputs = np.random.choice([-1, 1], size=(NUM_TRIES, model.num_neurons))
    #random_inputs = np.vstack((random_inputs, train_data))
    #random_inputs = train_data
    #NUM_TRIES = len(train_data)
    #random_inputs = train_data

    #print("random inputs", random_inputs)

    unique_local_mins = []

    num_training_patterns_found = 0

    for i in range(len(random_inputs)):
        model.recall(random_inputs[i], IS_SYNC, MAX_ITER)

        # Convert to a list if necessary, for consistency
        generated_local_min = np.array(model.neurons)

        # Check if the generated local minimum is unique using numpy's efficient array comparison
        if not any(np.array_equal(generated_local_min, local_min) for local_min in unique_local_mins):
            unique_local_mins.append(generated_local_min.tolist())

            #If matches a known pattern --> +1
            if any(np.array_equal(generated_local_min, train_min) for train_min in train_data):
                num_training_patterns_found += 1


    #print(unique_local_mins)
    #for i in range(len(unique_local_mins)):
       
        #print(unique_local_mins[i] * )


    print("Number of unique local minimas (attractors) found were", len(unique_local_mins))
    print(num_training_patterns_found, "of those were training patterns")

    return unique_local_mins



def scramble_data(data, percentage, mean, std):
    
    num_rows, num_cols = data.shape


    noisy_data = data.copy()

    for i in range(num_rows):
        # Calculate the number of elements to replace with noise
        num_noise_elements = int(np.ceil(percentage * num_cols))

        # Randomly choose indices to replace with noise
        noise_indices = np.random.choice(num_cols, num_noise_elements, replace=False)

        # Generate random binary noise (-1 or 1)
        #noise_values = np.random.choice([-1, 1], size=num_noise_elements)


        # Replace selected indices with noise values
        noisy_data[i, noise_indices] = -1 * noisy_data[i, noise_indices]

    return noisy_data





def visualize_img_path(image, folderpath, filename, save_to_file, plot_title=""):
    #alternative approach
    image = image*255

 
    plt.title(plot_title)
    plt.imshow(image)

    plt.gray()

    if save_to_file:
        
        if not os.path.isdir(folderpath):
            os.makedirs(folderpath)
        plt.savefig(folderpath + filename + ".png")
    else:
        plt.show()
