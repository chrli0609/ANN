import numpy as np
from hopfield import *
import matplotlib.pyplot as plt


def generate_img_array(filepath, img_dim):
    file = open(filepath, "r")

    pattern_list = file.readline().split(",", 8)

    file.close()

    #want 9 matrices that are 32x32
    img_array = np.zeros((9, img_dim, img_dim))
    for i in range(len(pattern_list)):
        img_array[i] = convert_1024_to_img_dim(pattern_list[i].split(","), img_dim)


def convert_1024_to_img_dim(vector, img_dim):
    img = np.zeros((img_dim, img_dim))
    for i in range(img_dim):

        for j in range(img_dim):
            img[i,j] = vector[i*img_dim+j]

    return img

def visualize_img(image):
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

    plt.show()





