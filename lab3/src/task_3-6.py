import numpy as np
from scipy.spatial.distance import hamming

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



from functions import *
from sparse_hopfield import Sparse_Hopfield


#Data stuff
MAX_NUM_PATTERNS_TO_TRY = 20
#NUM_TRAINING_PATTERNS = 10

#Model params
HAS_SELF_CONNECTION = False
NUM_NEURONS = 100
IS_SYNC = False
MAX_ITER = 5



#Outputs
SAVE_TO_FILE = True


hamming_per_num_memory_list = []

for NUM_TRAINING_PATTERNS in range(1, MAX_NUM_PATTERNS_TO_TRY):

    #Create binary data with 10% activation probability
    #all_data = np.random.choice(a=[0, 1], p=[0.9, 0.1], size=(NUM_TRAINING_PATTERNS, NUM_NEURONS))
    #all_data = np.random.choice(a=[0, 1], p=[0.95, 0.05], size=(NUM_TRAINING_PATTERNS, NUM_NEURONS))
    all_data = np.random.choice(a=[0, 1], p=[0.99, 0.01], size=(NUM_TRAINING_PATTERNS, NUM_NEURONS))

    #Compute rho
    rho = 0 
    for i in range(NUM_TRAINING_PATTERNS):
        for j in range(NUM_NEURONS):
            rho += all_data[i][j]

    rho = rho/(NUM_TRAINING_PATTERNS * NUM_NEURONS)

    #print("rho", rho)




    model = Sparse_Hopfield(NUM_NEURONS, HAS_SELF_CONNECTION)
    model.train(all_data, rho)



    #Iterate over recall for different values of THETA
    theta_list = np.arange(-1, 1.1, 0.25)
    #theta_list = np.arange(-1, 1.1, 0.5)
    #print(theta_list)

    hamming_score_per_theta_list = []
    for i in range(len(theta_list)):

        hamming_score_sum = 0
        #Check the quality of each recall
        for j in range(len(all_data)):
            model.recall(all_data[j], IS_SYNC, MAX_ITER, theta_list[i])


            hamming_score_sum += hamming(all_data[j], model.neurons)
            #print("hamming(all_data[j], model.neurons)", hamming(all_data[j], model.neurons))
            
        

        hamming_score_per_theta_list.append(round(hamming_score_sum/len(all_data), 4))



    #plt.plot(theta_list, hamming_score_per_theta_list)
    #plt.title("Hamming score vs Theta")
    #plt.show()

    hamming_per_num_memory_list.append(hamming_score_per_theta_list)







# Convert 2D list to a NumPy array
data_2d_np = np.array(hamming_per_num_memory_list)






from prettytable import PrettyTable
t = PrettyTable(['Num Mem', '-1', '-0.75', '-0.5', '-0.25', '0', '0.25', '0.5', '0.75', '1'])
#t = PrettyTable(['Num Mem', '-1', '-0.5', '0', '0.5', '1'])
for i in range(len(hamming_per_num_memory_list)):

    hamming_per_num_memory_list[i].insert(0, i+3)
    
    t.add_row(hamming_per_num_memory_list[i])
print(t)



prettytable_to_latex(t, "../out/task_3-6/latex_hd_rho_"+str(round(100*rho, 2))+".txt")




num_memories = []
for i in range(len(hamming_per_num_memory_list)):

    num_memories.append(i+3)


num_memories = np.linspace(3, 20, data_2d_np.shape[0])




# Create a meshgrid for X and Y
X, Y = np.meshgrid(theta_list, num_memories)




# Plotting the 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface (Z is the 2D array)
surf = ax.plot_surface(X, Y, data_2d_np, cmap='viridis')

# Labels
fig.suptitle("Average Hamming Distance over reconstructed patterns vs Num memories stored and Theta\nRho: " + str(round(rho,3)))
ax.set_ylabel('Num Memories')
ax.set_xlabel('Theta List')
ax.set_zlabel('Hamming Distance')


# Set Z-axis limits
ax.set_zlim(0.0, 1.0)

# Set Z-axis ticks to include 0.1 increments
ax.zaxis.set_ticks(np.arange(0.0, 1.1, 0.1))





# Show plot
plt.show()
