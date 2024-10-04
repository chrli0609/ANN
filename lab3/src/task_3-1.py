import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from scipy.spatial.distance import hamming



from functions import *
from hopfield import Hopfield

MAX_ITER = 15
HAS_SELF_CONNECTIONS = False
IS_SYNC = False





x1= [-1,-1, 1,-1, 1,-1,-1, 1]
x2= [-1,-1,-1,-1,-1, 1,-1,-1]
x3= [-1, 1, 1,-1,-1, 1,-1, 1]


X = np.vstack((x1, x2, x3))


model = Hopfield(len(x1), HAS_SELF_CONNECTIONS)


model.train(X)







print("==============================================================================")
print("####################### Evaluate Resistance to Noise #########################")
print("==============================================================================")



x1d = [ 1,-1, 1,-1, 1,-1,-1, 1]
x2d = [ 1, 1,-1,-1,-1, 1,-1,-1]
x3d = [ 1, 1, 1,-1, 1, 1,-1, 1]

xd = np.vstack((x1d, x2d, x3d))


pred_d = []
for i in range(len(xd)):
    model.recall(xd[i], IS_SYNC, MAX_ITER)
    pred_d.append(np.array(model.neurons))
    #print("Noisy input:\t", xd[i], "\t", np.count_nonzero(X[i]-xd[i]), "bit errors")
    #print("Clean input:\t", X[i])
    #print("Predicted:\t", pred_d[i])
    #print("acc score vs noisy", 1-np.count_nonzero(pred_d[i]-xd[i])/len(xd[i]))
    print("acc score vs clean", 1-np.count_nonzero(pred_d[i]-X[i])/len(X[i]))
    print("-------------------------------")


print("==============================================================================")
print("######################## Evaluate number of attractors #######################")
print("==============================================================================")

######################## Evaluate number of attractors ########################
NUM_TRIES = 10**4

get_attractors(model, X, IS_SYNC, MAX_ITER, NUM_TRIES)


print("==============================================================================")
print("################## Make starting pattern very different ######################")
print("==============================================================================")

############ Make starting pattern very different ################
xw1 = [ 1,-1,-1,-1,-1, 1, 1,-1]
xw2 = [ 1,-1, 1,-1, 1, 1, 1, 1]
xw3 = [ 1,-1,-1,-1, 1, 1, 1,-1]


xw = np.vstack((xw1, xw2, xw3))



pred_d = []
for i in range(len(xw)):
    model.recall(xw[i], IS_SYNC, MAX_ITER)
    pred_d.append(np.array(model.neurons))
    print("Noisy input:\t", xw[i], "\t", np.count_nonzero(X[i]-xw[i]), "bit errors")
    print("Clean input:\t", X[i])
    print("Predicted:\t", pred_d[i])
    #print("acc score vs noisy", 1-np.count_nonzero(pred_d[i]-xw[i])/len(xw[i]))
    print("acc score vs clean", 1-np.count_nonzero(pred_d[i]-X[i])/len(X[i]))
    print("-------------------------------")




print("==============================================================================")
print("################## Make Starting Patterns More Dissimilar ######################")
print("==============================================================================")


num_stored_patterns, num_neurons = X.shape

hamming_per_noise_lvl_per_pattern = []
for i in range(num_neurons+1):

    

    #print("scrambled\n", scrambled_data)
    #print("X\n", X)


    hamming_per_pattern = []
    #For each pattern do recall with scrambled data
    for j in range(num_stored_patterns):
        scrambled_pattern = flip_bits(X[j], i)
        model.recall(scrambled_pattern, IS_SYNC, MAX_ITER)

        hamming_per_pattern.append(hamming(model.neurons, X[j]))
    
    hamming_per_noise_lvl_per_pattern.append(hamming_per_pattern)





t = PrettyTable(['Num Bits Flipped', 'x1', 'x2', 'x3'])
for i in range(len(hamming_per_noise_lvl_per_pattern)):

    hamming_per_noise_lvl_per_pattern[i].insert(0, i)
    
    t.add_row(hamming_per_noise_lvl_per_pattern[i])
print(t)


prettytable_to_latex(t, "../out/task_3-1/latex_bits_flipped.txt")




np_hamming = np.array(hamming_per_noise_lvl_per_pattern)



plt.title("Hamming Distance of Recalled Pattern from Distorted Pattern")
plt.plot(np_hamming[:, 1], label='x1')
plt.plot(np_hamming[:, 2], label='x2')
plt.plot(np_hamming[:, 3], label='x3')

plt.xlabel("Number of Bit flips")
plt.ylabel("Hamming Distance of Memorized Pattern vs Recalled Pattern")


plt.legend()
plt.show()








#### Compute the energy level of each local 