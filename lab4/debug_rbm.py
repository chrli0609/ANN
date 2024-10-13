from util import *
from rbm import RestrictedBoltzmannMachine 
#from rbm_batch import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet
import matplotlib.pyplot as plt

N_TRAIN = 60000
N_TEST = 10000
BATCH_SIZE = 20
N_ITERATIONS = int(N_TRAIN/BATCH_SIZE* 1.2) 
N_EPOCHS = 20



if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=N_TRAIN, n_test=N_TEST)

    ''' restricted boltzmann machine '''
    
    print ("\nStarting a Restricted Boltzmann Machine..")


    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=4,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=BATCH_SIZE
    )
    
    



    #print("train_imgs\n", train_imgs)

    visualize_data(train_imgs)

    rbm.cd1(visible_trainset=train_imgs, n_iterations=N_ITERATIONS)


    plot_weight_change(rbm)

