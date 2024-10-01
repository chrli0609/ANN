import numpy as np
from hopfield import *
import matplotlib.pyplot as plt


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
    
    """
    rows, cols = image.shape
    for i in range(rows):

        for j in range(cols):
            if (image[rows,cols] == 1):
                image[rows,cols] == 255"""

    plt.imshow(image)

    plt.gray()

    if save_to_file:
        plt.savefig("../out/task_3-2/p" + str(pattern_num) + "/" + str(image_num)+".png")
    else:
        plt.show()





