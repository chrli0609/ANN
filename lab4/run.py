from util import *
from rbm import RestrictedBoltzmannMachine 
#from rbm_batch import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet
import matplotlib.pyplot as plt

N_TRAIN = 60000
N_TEST = 10000
BATCH_SIZE = 20
N_ITERATIONS = 11

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
    

    visualize_data(train_imgs, "out/training_data_sample.png")
    rbm.cd1(visible_trainset=train_imgs, n_iterations=N_ITERATIONS)
    #rbm.cd1(visible_trainset=train_imgs, n_iterations=N_ITERATIONS)

    plot_weight_change(rbm)
    plot_loss(rbm)

    plt.clf()

    plt.plot(rbm.delta_bias_v_norm, label="Delta bias v norm")
    plt.plot(rbm.delta_bias_h_norm, label="Delta bias h norm")

    plt.xlabel('Iteration')
    plt.ylabel('Norm of bias over time')
    plt.legend()
    plt.savefig("out/rbm/weights/delta_bias_v_h_norm.png")

    plt.clf()
    plt.cla()
    plt.close()
    
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

    plot_3d_array(dbn.label_values)
    
    dbn.recognize(test_imgs, test_lbls)

    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="rbms")
    


    ''' fine-tune wake-sleep training '''
    '''
    dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=N_ITERATIONS)

    dbn.recognize(train_imgs, train_lbls)
    
    dbn.recognize(test_imgs, test_lbls)
    
    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="dbn")
    '''