import numpy as np
import math
import matplotlib.pyplot as plt

class RBF():

    def __init__(self, mu_list, variance_list):
        self.n_rbf_units = len(mu_list)

        self.variance = np.array(variance_list)

        
        
        #self.mu = np.random.rand(n_rbf_units, in_dim)
        #self.mu = np.array([[-2,-2], [0,0], [2,2]])
        self.mu = np.array(mu_list)



        self.w = np.random.rand(self.n_rbf_units, 1)
        

    
    def phi(self, i, x_vec):
        exponent = -(np.matmul((x_vec-self.mu[i]).T, (x_vec-self.mu[i]))) / (2 * self.variance[i])

        return np.power(math.e, exponent)

    def gen_phi_mat_vector(self, x_vec):
        N_samples, in_dim = x_vec.shape


        PHI = np.zeros((N_samples, self.n_rbf_units))

        for i in range(N_samples):
            for j in range(self.n_rbf_units):
                PHI[i,j] = self.phi(j, x_vec[i])
        
        return PHI


    def forward(self, X):

        PHI = self.gen_phi_mat_vector(X)

        return np.matmul(PHI, self.w)


    
    def batch_supervised_training(self, X, T):
        PHI = self.gen_phi_mat_vector(X)


        print("PHI", PHI.shape)
        print("PHI.T * PHI", np.matmul(PHI.T, PHI))
        print("np.linalg.inv(np.dot(PHI.T,PHI)", np.linalg.inv(np.dot(PHI.T,PHI)).shape)


        #Solve Least Squared System PHI.T * PHI * w = PHI.T * T
        self.w = np.dot((np.dot(np.linalg.inv(np.dot(PHI.T,PHI)),PHI.T)), T)
        
        
    

    def plot_2d_weight_space(self, x_range, granularity):
        x_vals_1 = np.linspace(x_range[0], x_range[1], granularity)
        x_vals_2 = np.linspace(x_range[0], x_range[1], granularity)
        X, Y = np.meshgrid(x_vals_1, x_vals_2)
        Z = np.zeros_like(X)
        
        plt.figure(figsize=(8, 6))
        for k in range(self.n_rbf_units):
            #Z = np.zeros_like(X)
            for i in range(len(x_vals_1)):
                for j in range(len(x_vals_2)):
                    Z[i, j] += self.phi(k, [x_vals_1[i], x_vals_2[j]])
        
        # Plot each contour plot, ensuring all are visible
        plt.contourf(X, Y, Z, cmap='viridis')
        
        
        plt.colorbar(label='RBF Output')
        plt.title("Radial Basis Function Response (2D)")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()
    



    def plot_rbf(self, x_range):
        """Plot the RBFs for a range of inputs."""
        plt.figure(figsize=(8, 6))

        # Create a grid of x values to evaluate
        x_vals = np.linspace(x_range[0], x_range[1], 100)
        y_vals = np.zeros_like(x_vals)

        # Evaluate each RBF and sum over units
        for i in range(self.n_rbf_units):
            for j, x in enumerate(x_vals):
                x_vec = np.array([x])  # Single input point
                y_vals[j] += self.w[i] * self.phi(i, x_vec)
        
        # Plot the summed RBF response
        plt.plot(x_vals, y_vals, label='RBF Response')
        plt.title("Radial Basis Function Response")
        plt.xlabel("x")
        plt.ylabel("RBF Output")
        plt.legend()
        plt.show()






def sine_func(x):
    return np.sin(2*x)

def square_func(x):
    return 1 if sine_func(x) > 0 else -1



    
# Plot the summed RBF response
STEP_LENGTH = 0.1
#n_rbf_units = 4
in_dim = 2
'''
mu_list = [
    [-4,-4],
    [-4, 4],
    [-2,-2],
    [-2, 2],
    [0, 0],
    [0,-4],
    [4, 0],
    [-4,0],
    [0, 4],
    [2, 2],
    [2,-2],
    [4,-4],
    [4, 4]
]

variance_list = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
'''

mu_list = np.arange(0, 2*np.pi, 0.3)
variance_list = [0.1] *len(mu_list)

"""
mu_list = []
for i in range(-4, 4, 4):
    for j in range(-4, 4, 4):
        mu_list.append([i,j])

variance_list = [0.5] * (len(mu_list))
"""

print("mu_list", mu_list)
print("variance_list", variance_list)

def generate_data(x_end, step_length, function):

    train_X = ((np.arange(0 , x_end, step_length)).T).reshape(-1,1)
    test_X = ((np.arange(0.05, x_end, step_length)).T).reshape(-1,1)
    

    N_train_samples = len(train_X)
    N_test_samples = len(test_X)



    train_F = np.zeros((N_train_samples, 1))
    test_F = np.zeros((N_test_samples, 1))

    for i in range(len(train_X)):
        train_F[i] = function(train_X[i])

    for i in range(len(test_X)):
        test_F[i] = function(test_X[i])

    print("train_X", train_X.shape)

    return train_X, train_F, test_X, test_F





#Generate Data for sine and square wave
sine_train_X, sine_train_F, sine_test_X, sine_test_F = generate_data(2*np.pi, STEP_LENGTH, sine_func)
#plt.plot(sine_train_X, sine_train_F , label="Sin(2x) training")
#plt.plot(sine_test_X, sine_test_F , label="Sin(2x) testing")
#plt.show()



# Sine
rbf_network = RBF(mu_list, variance_list)
rbf_network.plot_2d_weight_space((-5, 5), 100)

rbf_network.batch_supervised_training(sine_train_X, sine_train_F)

sine_pred_F = rbf_network.forward(sine_test_X)

plt.plot(sine_pred_F, label="Predicted Testing Data")
plt.plot(sine_train_F, label="Training Data")
plt.plot(sine_test_F, label="Testing Data")
plt.legend()
plt.show()



