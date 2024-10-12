from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet


N_TRAIN = 60000
N_TEST = 10000
#N_ITERATIONS = 6000
BATCH_SIZE = 1000
N_ITERATIONS = int(N_TRAIN/BATCH_SIZE)

if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=N_TRAIN, n_test=N_TEST)

    ''' restricted boltzmann machine '''
    
    print ("\nStarting a Restricted Boltzmann Machine..")

    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=200,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=BATCH_SIZE
    )
    
    rbm.cd1(visible_trainset=train_imgs, n_iterations=N_ITERATIONS)
    rbm.cd1(visible_trainset=train_imgs, n_iterations=N_ITERATIONS)

    plt.plot(rbm.debug_weights, color="blue")
    plt.title("Delta W for each sample")
    plt.xlabel("Sample")
    plt.ylabel("Delta W")
    #plt.legend()
    plt.savefig("out/rbm/weight_vs_it_batch_size_" + str(rbm.batch_size) + ".png")
    #plt.show()

    
    ''' deep- belief net '''

    print ("\nStarting a Deep Belief Net..")
    
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

    ''' fine-tune wake-sleep training '''

    dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=N_ITERATIONS)

    dbn.recognize(train_imgs, train_lbls)
    
    dbn.recognize(test_imgs, test_lbls)
    
    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="dbn")
