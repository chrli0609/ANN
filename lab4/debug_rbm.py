from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet
import matplotlib.pyplot as plt

N_TRAIN = 1000

if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=N_TRAIN, n_test=100)

    ''' restricted boltzmann machine '''
    
    print ("\nStarting a Restricted Boltzmann Machine..")

    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=500,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=10
    )
    
    rbm.cd1(visible_trainset=train_imgs, n_iterations=100)

    plt.plot(rbm.debug_weights, color="blue")
    plt.title("Delta W for each sample")
    plt.xlabel("Sample")
    plt.ylabel("Delta W")
    #plt.legend()
    plt.savefig("out/rbm/weight_vs_it_batch_size_" + str(rbm.batch_size) + ".png")
    #plt.show()
