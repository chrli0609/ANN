import numpy as np
import matplotlib.pyplot as plt
from hopfield import Hopfield



x1= [-1,-1, 1,-1, 1,-1,-1, 1]
x2= [-1,-1,-1,-1,-1, 1,-1,-1]
x3= [-1, 1, 1,-1,-1, 1,-1, 1]


X = np.vstack((x1, x2, x3))


model = Hopfield(len(x1), False)


model.synchronous_training(X)







print("==============================================================================")
print("####################### Evaluate Resistance to Noise #########################")
print("==============================================================================")



x1d = [ 1,-1, 1,-1, 1,-1,-1, 1]
x2d = [ 1, 1,-1,-1,-1, 1,-1,-1]
x3d = [ 1, 1, 1,-1, 1, 1,-1, 1]

xd = np.vstack((x1d, x2d, x3d))


pred_d = []
for i in range(len(xd)):
    pred_d.append(model.recall(xd[i]))
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
NUM_TRIES = 10**5

random_inputs = np.random.random((NUM_TRIES, len(x1)))

unique_local_mins = []
for i in range(NUM_TRIES):
    generated_local_min = model.recall(random_inputs[i])

    # Convert to a list if necessary, for consistency
    generated_local_min = generated_local_min.tolist()

    # Check if the generated local minimum is unique using numpy's efficient array comparison
    if not any(np.array_equal(generated_local_min, local_min) for local_min in unique_local_mins):
        unique_local_mins.append(generated_local_min)

print(np.array(unique_local_mins))
print("Number of unique local minimas (attractors) found were", len(unique_local_mins))


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
    pred_d.append(model.recall(xw[i]))
    print("Noisy input:\t", xw[i], "\t", np.count_nonzero(X[i]-xw[i]), "bit errors")
    print("Clean input:\t", X[i])
    print("Predicted:\t", pred_d[i])
    print("acc score vs noisy", 1-np.count_nonzero(pred_d[i]-xw[i])/len(xw[i]))
    print("acc score vs clean", 1-np.count_nonzero(pred_d[i]-X[i])/len(X[i]))
    print("-------------------------------")

