from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet

import numpy as np


N_TRAIN = 60000
N_TEST = 10000
#N_ITERATIONS = 1000
N_ITERATIONS = int(N_TRAIN/10)

#N_TRAIN = 60000
#N_TEST = 10000
#N_ITERATIONS = int(N_TRAIN/10)
#N_ITERATIONS = 10000


if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=N_TRAIN, n_test=N_TEST)

    ''' deep- belief net '''


    print("\nStarting a Deep Belief Net..")
    
    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=10
    )
    
    ''' greedy layer-wise training '''

    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=N_ITERATIONS)

    dbn.recognize(train_imgs, train_lbls)
    
    dbn.recognize(test_imgs, test_lbls)

    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="rbms")
