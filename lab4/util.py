import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#from run import N_TRAIN


#INDICES = np.random.choice(N_TRAIN, 25, replace=False)


def sigmoid(support):
    
    """ 
    Sigmoid activation function that finds probabilities to turn ON each unit. 
        
    Args:
      support: shape is (size of mini-batch, size of layer)      
    Returns:
      on_probabilities: shape is (size of mini-batch, size of layer)      
    """
    #print("support", support)
    on_probabilities = 1./(1.+np.exp(-support))
    return on_probabilities

def softmax(support):

    """ 
    Softmax activation function that finds probabilities of each category
        
    Args:
      support: shape is (size of mini-batch, number of categories)      
    Returns:
      probabilities: shape is (size of mini-batch, number of categories)      
    """

    expsup = np.exp(support-np.max(support,axis=1)[:,None])
    return expsup / np.sum(expsup,axis=1)[:,None]

def sample_binary(on_probabilities):    

    """ 
    Sample activations ON=1 (OFF=0) from probabilities sigmoid probabilities
        
    Args:
      support: shape is (size of mini-batch, size of layer)      
    Returns:
      activations: shape is (size of mini-batch, size of layer)      
    """
    #activations = 1. * ( on_probabilities >= 0.5 )
    activations = 1. * ( on_probabilities >= np.random.random_sample(size=on_probabilities.shape) )
    return activations

def sample_categorical(probabilities):

    """ 
    Sample one-hot activations from categorical probabilities
        
    Args:
      support: shape is (size of mini-batch, number of categories)      
    Returns:
      activations: shape is (size of mini-batch, number of categories)      
    """
    
    cumsum = np.cumsum(probabilities,axis=1)
    rand = np.random.random_sample(size=probabilities.shape[0])[:,None]    
    activations = np.zeros(probabilities.shape)
    activations[range(probabilities.shape[0]),np.argmax((cumsum >= rand),axis=1)] = 1
    return activations

def load_idxfile(filename):

    """
    Load idx file format. For more information : http://yann.lecun.com/exdb/mnist/ 
    """
    import struct
        
    with open(filename,'rb') as _file:
        if ord(_file.read(1)) != 0 or ord(_file.read(1)) != 0 :
           raise Exception('Invalid idx file: unexpected magic number!')
        dtype,ndim = ord(_file.read(1)),ord(_file.read(1))
        shape = [struct.unpack(">I", _file.read(4))[0] for _ in range(ndim)]
        data = np.fromfile(_file, dtype=np.dtype(np.uint8).newbyteorder('>')).reshape(shape)
    return data
    
def read_mnist(dim=[28,28],n_train=60000,n_test=1000):

    """
    Read mnist train and test data. Images are normalized to be in range [0,1]. Labels are one-hot coded.
    """    

    train_imgs = load_idxfile("train-images-idx3-ubyte")
    train_imgs = train_imgs / 255.
    train_imgs = train_imgs.reshape(-1,dim[0]*dim[1])

    train_lbls = load_idxfile("train-labels-idx1-ubyte")
    train_lbls_1hot = np.zeros((len(train_lbls),10),dtype=np.float32)
    train_lbls_1hot[range(len(train_lbls)),train_lbls] = 1.

    test_imgs = load_idxfile("t10k-images-idx3-ubyte")
    test_imgs = test_imgs / 255.
    test_imgs = test_imgs.reshape(-1,dim[0]*dim[1])

    test_lbls = load_idxfile("t10k-labels-idx1-ubyte")
    test_lbls_1hot = np.zeros((len(test_lbls),10),dtype=np.float32)
    test_lbls_1hot[range(len(test_lbls)),test_lbls] = 1.

    return train_imgs[:n_train],train_lbls_1hot[:n_train],test_imgs[:n_test],test_lbls_1hot[:n_test]

def viz_rf(weights,it,grid):

    """
    Visualize receptive fields and save 
    """
    fig, axs = plt.subplots(grid[0],grid[1],figsize=(grid[1],grid[0]))#,constrained_layout=True)
    plt.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=0,hspace=0)        
    imax = abs(weights).max()
    for x in range(grid[0]):
        for y in range(grid[1]):
            axs[x,y].set_xticks([]);
            axs[x,y].set_yticks([]);
            axs[x,y].imshow(weights[:,:,y+grid[1]*x], cmap="bwr", vmin=-imax, vmax=imax, interpolation=None)
    plt.savefig("out/rbm/viz_rf/rf.iter%06d.png"%it)
    plt.close('all')

def stitch_video(fig,imgs):
    """
    Stitches a list of images and returns a animation object
    """
    import matplotlib.animation as animation
    
    return animation.ArtistAnimation(fig, imgs, interval=100, blit=True, repeat=False)    


def plot_weight_change(rbm):
    plt.plot(rbm.debug_delta_weights, color="blue")
    plt.title("Norm of delta W for each minibatch update")
    plt.xlabel("Minibatch")
    plt.ylabel("Norm of delta W")
    #plt.legend()
    plt.savefig("out/rbm/weights/delta_weight_vs_it_batch_size_" + str(rbm.batch_size) + ".png")
    #plt.show()

    plt.clf()
    plt.cla()
    plt.close()

    plt.plot(rbm.debug_weights, color="blue")
    plt.title("Norm of weight values for each sample")
    plt.xlabel("Minibatch")
    plt.ylabel("Norm of Weight values")
    #plt.legend()
    plt.savefig("out/rbm/weights/weight_vs_it_batch_size_" + str(rbm.batch_size) + ".png")


    plt.clf()
    plt.cla()
    plt.close()


def visualize_data(data, filepath):
    """
    Randomly selects 25 rows from a 2D numpy array where each row has 784 columns 
    and reshapes each into a 28x28 image. The function also plots the 25 images.
    
    Args:
    data (numpy.ndarray): A 2D numpy array of shape (n, 784), where n is the number of rows.
    
    Returns:
    numpy.ndarray: A 3D numpy array of shape (25, 28, 28) containing the reshaped images.
    """
    if data.shape[1] != 784:
        raise ValueError("Each row in the input array must have 784 columns.", data.shape[1], "found")
    
    # Randomly select 25 rows
    #indices = np.random.choice(data.shape[0], 25, replace=False)
    INDICES = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

    indices = INDICES
    
    # Reshape each row into a 28x28 image
    images = data[indices].reshape(25, 28, 28)
    
    # Plot the images in a 5x5 grid
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')  # Hide axes for clarity
    
    plt.tight_layout()
    plt.savefig(filepath)
    plt.clf()
    plt.cla()
    plt.close()
    #plt.show()

def plot_loss(rbm):
    plt.plot(rbm.losses)
    plt.title("MSE Loss vs epoch")
    plt.xlabel("Weight Update from each Epoch")
    plt.ylabel("Mean Weight values")
    #plt.legend()
    plt.savefig("out/rbm/loss/" + "mse_loss_"+str(rbm.batch_size) + ".png")

def plot_3d_array(array):
    array=array.T
    # Get array dimensions
    rows, cols = array.shape
    
    # Create meshgrid for x and y axes
    x, y = np.meshgrid(range(cols), range(rows))
    
    # Get values for the z-axis from the array
    z = array
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot a surface plot using the array values as the z coordinates
    ax.plot_surface(x, y, z, cmap='viridis')
    
    # Set axis labels
    ax.set_ylabel('Labels')
    ax.set_xlabel('Step')
    ax.set_zlabel('Probability')
    
    # Set plot title
    ax.set_title('Probabilities of labels for an image during recognition')
    
    plt.show()
    plt.savefig("out/dbn/label_values.png")