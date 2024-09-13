import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import PolyCollection
from matplotlib.animation import FuncAnimation


def generate_color_list(T):

    
    color_list = []
    for i in range(len(T)):
        if T[i] == 1:
            color_list.append("red")
        elif T[i] == -1:
            color_list.append("blue")
        else:
            print("Error found in generated Target list")
    

    return color_list



def gen_non_lin_data(in_dim, ndata, mA, mB, sigmaA, sigmaB):
    #ndata = 100

    classA = np.zeros((2, ndata))
    classB = np.zeros((2, ndata))

    classA[0, :] = np.concatenate([
    np.random.randn(round(0.5 * ndata)) * sigmaA - mA[0],
    np.random.randn(round(0.5 * ndata)) * sigmaA + mA[0]
    ])
    classA[1, :] = np.random.randn(ndata) * sigmaA + mA[1]

    classB[0, :] = np.random.randn(ndata) * sigmaB + mB[0]
    classB[1, :] = np.random.randn(ndata) * sigmaB + mB[1]



    
    
    X = np.concatenate((classA, classB), axis=1)
    



    return X


def subsampling_25_from_each_class(all_data):

    num_rows, num_cols = all_data.shape
    size = int(0.75*num_cols/2)

    classA = all_data[:, :int(num_cols/2)]
    classB = all_data[:, int(num_cols/2):]


    np.random.shuffle(np.transpose(classA))
    np.random.shuffle(np.transpose(classB))

    
    train_A = classA[:,:size]
    train_B = classB[:,:size]

    test_A = classA[:,size:]
    test_B = classB[:,size:]

    

    data_train = np.concatenate((train_A, train_B), axis=1)
    data_test = np.concatenate((test_A, test_B), axis=1)

    
    np.random.shuffle(np.transpose(data_train))
    np.random.shuffle(np.transpose(data_test))





    return data_train, data_test



def subsampling_50_from_classA(all_data):

    num_rows, num_cols = all_data.shape
    size = int(0.75*num_cols/2)

    classA = all_data[:, :int(num_cols/2)]
    classB = all_data[:, int(num_cols/2):]

 
    np.random.shuffle(np.transpose(classA))
    
    train_A = classA[:,:size]
    test_A = classA[:,size:]
    

    data_train = np.concatenate((train_A, classB), axis=1)
    data_test = test_A

    
    np.random.shuffle(np.transpose(data_train))
    np.random.shuffle(np.transpose(data_test))



    return data_train, data_test





    

#20% from a subset of classA for which classA(1,:)<0 and 80% from a
#subset of classA for which classA(1,:)>0
def subsampling_point_2_lt_0_and_point_8_gt_0_from_A(all_data):


    num_rows, num_cols = all_data.shape
    size_group1 = int(0.8*num_cols/2)
    size_group2 = int(0.2*num_cols/2)

    classA = all_data[:, :int(num_cols/2)]
    classB = all_data[:, int(num_cols/2):]


    group1_mask = classA[0,:] < 0
    group2_mask = classA[0,:] > 0


    group1_list = classA[:,group1_mask]
    group2_list = classB[:,group2_mask]

    #Randomize within group 1 and group 2
    np.random.shuffle(np.transpose(group1_list))
    np.random.shuffle(np.transpose(group2_list))


    #Take 80% of group1 for training
    train_group1 = group1_list[:,:size_group1]
    #Take 20% of group1 for testing
    test_group1 = group1_list[:,size_group1:]

    #Take 20% of group2 for training
    train_group2 = group2_list[:,:size_group2]
    #Take 80% of group2 for testing
    test_group2 = group2_list[:, size_group2:]
    
    

    data_train = np.concatenate((train_group1, train_group2, classB), axis=1)
    data_test = np.concatenate((test_group1, test_group2), axis=1)

    
    np.random.shuffle(np.transpose(data_train))
    np.random.shuffle(np.transpose(data_test))



    return data_train, data_test



def generate_random_non_linear_input_and_weights(in_dim, n, mA, mB, sigmaA, sigmaB, subsampling_function):


    X = gen_non_lin_data(in_dim, n, mA, mB, sigmaA, sigmaB)



    #Target matrix
    t1 = np.ones(n)
    t2 =-np.ones(n)
    T = np.concatenate((t1, t2))
    

    
    #Split some parts of the data for testing
    #split_index = int(train_test_dist * (2*n))
    
    
    #Merge both answer and input into the same matrix
    all_data = np.concatenate((X, T[:, None].T), axis=0)


    training_set, test_set = subsampling_function(all_data)
    
    
    #Test input and test target
    X_test = test_set[:in_dim,:]
    T_test = test_set[-1,:]
    
    
    
    ####### Generate the input X and target T to be used during training #######
    X = training_set[:in_dim,:]
    T = training_set[-1, :]
    


    #Generate color list
    color_list = generate_color_list(T)
    color_list_test = generate_color_list(T_test)
    
    
    #Add extra row at bottom for bias
    W = np.random.rand(in_dim+1, 1)


    print("W:", W.shape)
    print("X:", X.shape)
    print("T:", T.shape)
    print("X_text:", X_test.shape)
    print("T_test:", T_test.shape)
    

    return W, X, T, X_test, T_test, color_list, color_list_test






def single_to_double_T(T):
    
    multi_T = []
    for target in T:
        expected_out = [None] * 2
        if target == 1:
            expected_out = [1, 0]
        elif target == -1:
            expected_out = [0, 1]
        else:
            print("An error has been found in Target Vector")

        multi_T.append(expected_out)

    multi_T_np = np.array(multi_T)

    return multi_T_np.T




def max_of_col(O):
    _, num_cols = O.shape

    new_O = []
    for i in range(num_cols):
        each_row = [None] * 2
        if O[0][i] > O[1][i]:
             each_row[0] = 1
             each_row[1] = 0
        else:
            each_row[0] = 0
            each_row[1] = 1
        new_O.append(each_row)

    return np.array(new_O).T


def sign(num):
    return -1 if num < 0 else 1

def accuracy_score(O, T):

    correct = 0
    incorrect = 0
    for i in range(len(O)):
        if sign(O[i]) == sign(T[i]):
            correct += 1
        else:
            incorrect += 1
    
    return correct / (correct + incorrect)
            




def plot_data(X, color_list, X_test, color_list_test):

	fig, ax = plt.subplots(nrows=2, ncols=1)


	plt.subplot(2, 1, 1)
	plt.scatter(X[0,:], X[1,:], c=color_list, label="Training data")
	plt.legend(loc="upper right")

	plt.subplot(2, 1, 2)
	plt.scatter(X_test[0,:], X_test[1,:], c=color_list_test, label="Validation data")
	plt.legend(loc="upper right")

	plt.show()



def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]

def three_d_plot(my_vec):
    # Parameterized dimensions of my_vec
    num_x_points = 64  # Number of points along the x-axis
    num_lambdas = 20   # Number of lambda values (or "slices" in y)

    # Set up the figure and 3D axis
    ax = plt.figure().add_subplot(projection='3d')

    # Generate x values based on the number of points
    x = np.linspace(0., num_x_points, num_x_points)

    # Assume my_vec is of shape (num_x_points, num_lambdas)
    #my_vec = np.random.rand(num_x_points, num_lambdas)  # Replace with actual data if available

    my_vec = np.array(my_vec)

    # verts[i] is a list of (x, y) pairs for each lambda value
    verts = [polygon_under_graph(x, my_vec[:, i]) for i in range(num_lambdas)]

    # Using lambda values corresponding to the number of lambdas
    lambdas = range(1, num_lambdas + 1)

    # Define facecolors
    facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(verts)))

    # Create the PolyCollection object
    poly = PolyCollection(verts, facecolors=facecolors, alpha=1.0)  # Opaque polygons

    ax.add_collection3d(poly, zs=lambdas, zdir='y')

    # Set limits and labels (adjust zlim based on data range)
    ax.set(xlim=(0, num_x_points), ylim=(1, num_lambdas + 1), zlim=(0, 5),
        xlabel='Epoch Number', ylabel=r'Number of Neurons in Hidden Layer', zlabel='Mean Squared Error')

    plt.show()




# Animation function
def animate_training_validation_errors(num_epochs, hidden_nodes_list, train_error_hidden_nodes_list, valid_error_hidden_nodes_list, subsampling_method, LEARNING_RATE, OUT_FOLDER):
    # Convert errors to arrays for easy indexing
    train_error_hidden_nodes_list = np.array(train_error_hidden_nodes_list)
    valid_error_hidden_nodes_list = np.array(valid_error_hidden_nodes_list)



    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Initialize lines for training and validation error
    train_line, = ax.plot([], [], label='Training Error', color='blue')
    val_line, = ax.plot([], [], label='Validation Error', color='red')

    # Set axis limits
    ax.set_xlim(1, num_epochs)
    ax.set_ylim(0, 2)  # Adjust based on your error scale
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error')
    title_str = subsampling_method.__name__ + '\nTraining and Validation Errors Across Epochs'
    ax.set_title(title_str)
    ax.legend()

    # Add grid
    ax.grid(True)

    # Set x and y ticks
    ax.set_xticks(np.arange(1, num_epochs + 1, 1))
    ax.set_yticks(np.linspace(0, 2.1, num=22))

    # Create text annotation for neuron count
    neuron_text = ax.text(0.8, 0.9, '', transform=ax.transAxes, fontsize=12, color='black',
                          bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    # Function to initialize the plot
    def init():
        train_line.set_data([], [])
        val_line.set_data([], [])
        neuron_text.set_text('')
        return train_line, val_line, neuron_text

    # Function to update the plot for each frame in the animation
    def update(frame):
        neurons = hidden_nodes_list[frame]  # Current number of neurons

        # Update training and validation lines with the corresponding errors
        train_line.set_data(np.arange(1, num_epochs + 1), train_error_hidden_nodes_list[frame])
        val_line.set_data(np.arange(1, num_epochs + 1), valid_error_hidden_nodes_list[frame])

        # Update the title to reflect the current neuron count
        ax.set_title(subsampling_method.__name__ + f'\nTraining and Validation Errors with $\eta$: '+str(LEARNING_RATE))

        # Update the neuron count text annotation
        neuron_text.set_text(f'Neurons: {neurons}')

        return train_line, val_line, neuron_text

     # Create the animation
    ani = FuncAnimation(fig, update, frames=len(hidden_nodes_list), init_func=init, blit=True, interval=600)

    # Save the animation as a gif
    ani.save(OUT_FOLDER + subsampling_method.__name__ + '_error.gif', writer='pillow')
    """def save_gifs():
        for i in range(len(train_error_hidden_nodes_list)):
            train_artists.append(train_container)
            valid_artists.append(valid_container)

        ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=600)
        plt.show(block=False)
        f = OUT_FOLDER + filepath_str + '.gif'
        writergif = animation.PillowWriter(fps=5)
        ani.save(f, writer=writergif)"""


    # Create the animation
    #ani = FuncAnimation(fig, update, frames=len(hidden_nodes_list), init_func=init, blit=True, interval=600)

    # Show the animation
    plt.show()


