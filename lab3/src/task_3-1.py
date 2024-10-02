import numpy as np
import matplotlib.pyplot as plt


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
    print("Noisy input:\t", xd[i], "\t", np.count_nonzero(X[i]-xd[i]), "bit errors")
    print("Clean input:\t", X[i])
    print("Predicted:\t", pred_d[i])
    print("acc score vs noisy", 1-np.count_nonzero(pred_d[i]-xd[i])/len(xd[i]))
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
    print("acc score vs noisy", 1-np.count_nonzero(pred_d[i]-xw[i])/len(xw[i]))
    print("acc score vs clean", 1-np.count_nonzero(pred_d[i]-X[i])/len(X[i]))
    print("-------------------------------")

