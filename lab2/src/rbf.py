import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class RBF():

    def __init__(self, mu_list, variance_list):
        self.n_rbf_units = len(mu_list)

        self.variance = np.array(variance_list)

        
        
        #self.mu = np.random.rand(n_rbf_units, in_dim)
        #self.mu = np.array([[-2,-2], [0,0], [2,2]])
        self.mu = np.array(mu_list)



        self.w = np.random.rand(self.n_rbf_units, 1)
        

    
    def phi(self, i, x_vec):
        # Ensure x_vec and mu[i] are treated as 1D arrays
        x_vec = np.atleast_1d(x_vec)
        mu_i = np.atleast_1d(self.mu[i])

        # Calculate the exponent for the RBF
        exponent = -(np.dot((x_vec - mu_i).T, (x_vec - mu_i))) / (2 * self.variance[i])
        return np.exp(exponent)

    def gen_phi_mat_vector(self, x_vec):
        # Assuming x_vec is 1D and self.mu contains the centers
        num_data_points = len(x_vec)
        num_centers = len(self.mu)  # Number of RBF centers

        # Initialize PHI matrix
        PHI = np.zeros((num_data_points, num_centers))

        # Compute PHI matrix
        for i in range(num_data_points):
            for j in range(num_centers):
                PHI[i, j] = self.phi(j, x_vec[i])

        return PHI
    
    def gen_seq_phi_mat(self, x_single_sample):
        num_centers = len(self.mu)  # Number of RBF centers

        # Initialize PHI matrix
        PHI = np.zeros((num_centers, 1))

        for i in range(num_centers):
            PHI[i] = self.phi(i, x_single_sample)
        
        return PHI




    def forward(self, X):

        PHI = self.gen_phi_mat_vector(X)

        return np.matmul(PHI, self.w)




    """def competitive_learning(self, X, F, lr, num_epochs):


        for epoch in range(num_epochs):
            
            self.plot_centroids((0,7),100)

            #Shuffle the order of the data
            indices = np.random.permutation(len(X))
            X = X[indices]

            for i in range(len(X)):
                #Compute distance between input and each centroid
                d_vec = np.linalg.norm(X[i]-self.mu)

                #Find the closest centroid
                max_index = np.argmax(d_vec)

                #Update that weight
                self.mu[max_index] += lr * (X[i]-self.mu[max_index])"""


    def competitive_learning(self, X, lr, num_epochs):
        
        fig, ax = plt.subplots()

        # Scatter plot to visualize centroids and input data points (1D)
        scatter_centroids, = ax.plot(self.mu, np.zeros_like(self.mu), 'ro', label="Centroids")
        scatter_data, = ax.plot(X, np.zeros_like(X), 'bx', alpha=0.5, label="Input Data")

        # Set limits for the plot based on input data
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(-1, 2)  # Fixed y-range since data is 1D

        # Function to update the plot for each frame
        def update(epoch):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            #X_shuffled = X

            print("mu", self.mu)

            for i in range(len(X_shuffled)):
                # Compute distance between input and each centroid (1D case)
                d_vec = []
                for j in range(len(self.mu)):
                    #print("X[i]", X[i])
                    #print("mu[j]", self.mu[j])
                    d_vec.append(np.linalg.norm(X_shuffled[i] - self.mu[j]))
                    #d_vec.append(abs((X[i] - self.mu[j])))

                print("X[i]", X[i])
                # Find the closest centroid
                min_index = np.argmin(d_vec)
                print("min_index", min_index)
                print("X_shuffled[i] - self.mu[min_index]", X_shuffled[i] - self.mu[min_index])

                # Update the corresponding weight
                self.mu[min_index] += lr * (X_shuffled[i] - self.mu[min_index])
                #self.mu[min_index] += lr * X_shuffled[i]
                print("mu", self.mu)

            # Update scatter plot data for centroids
            scatter_centroids.set_xdata(self.mu.flatten())

            ax.set_title(f'Epoch: {epoch + 1}/{num_epochs}')
            return scatter_centroids, scatter_data

        # Create the animation
        anim = FuncAnimation(fig, update, frames=num_epochs, repeat=False, interval=200)

        plt.legend()
        plt.show()


    def batch_supervised_training(self, X, T):
        PHI = self.gen_phi_mat_vector(X)


        print("PHI", PHI.shape)
        print("PHI.T * PHI", np.matmul(PHI.T, PHI))
        print("np.linalg.inv(np.dot(PHI.T,PHI)", np.linalg.inv(np.dot(PHI.T,PHI)).shape)


        #Solve Least Squared System PHI.T * PHI * w = PHI.T * T
        self.w = np.dot((np.dot(np.linalg.inv(np.dot(PHI.T,PHI)),PHI.T)), T)



    def seq_delta_training(self, X, T, lr, num_epochs):
        
        #For each epoch
        for epoch in range(num_epochs):

            #Shuffle data indices
            indices = np.random.permutation(len(X))
            X = X[indices]
            T = T[indices]

            #Loop over all the data points
            for i in range(len(X)):
                
                PHI =self.gen_seq_phi_mat(X[i])

                # PHI.T (1 x n_rbf_units)
                # w     (n_rbf_units x 1)
                # PHI   (n_rbf_units x 1)

                err = T[i] - np.matmul(PHI.T, self.w) 
                delta_w = lr*err*PHI

                self.w += delta_w
                

        



    def abs_res_err(self, F_pred, F_test):
        return np.mean(abs(F_pred-F_test))

        
        
    def plot_rbf_1d_inputs(self, x_range, granularity, test_X, pred_F, test_F, wave_type):
        """Plot the 1D position of RBF units in the weight space along with their distributions."""

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))  # Create a figure with two subplots
        


        # Add a supertitle for the entire figure
        fig.suptitle(f"Absolute Residual Error: {round(self.abs_res_err(pred_F, test_F),4)}", fontsize=16)


        # First subplot: 1D RBF Unit Distributions in Weight Space
        ax[0].set_title("1D RBF Unit Distributions in Weight Space")
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("Weighted RBF Output")
        ax[0].grid(True)

        # Add text showing how many RBF units we have
        num_rbf_units = self.n_rbf_units
        ax[0].text(0.75, 0.95, f'Number of RBF Units: {num_rbf_units}\n$\sigma^2$: {self.variance[0]}', transform=ax[0].transAxes, 
               fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))


        x_vals = np.linspace(x_range[0], x_range[1], granularity)  # X values for plotting the RBF curves

        # Plot each RBF unit's distribution
        for i in range(self.n_rbf_units):
            y_vals = []
            for x in x_vals:
                x_vec = np.array([x])  # Ensure x_vec has shape (n, 1)
                # Calculate the RBF output for the current unit
                rbf_value = self.w[i] * self.phi(i, x_vec)  # Weighted RBF output
                y_vals.append(rbf_value)

            # Plot the RBF distribution as a curve
            ax[0].plot(x_vals, y_vals)#, label=f'RBF Unit {i+1} (μ={self.mu[i]}, σ={self.variance[i]})')

            # Mark the center position (mu) and weight (w) with a scatter point
            ax[0].scatter(self.mu[i], self.w[i], color='red', zorder=5)  # Higher zorder to appear on top



        ax[0].legend()  # Add a legend for the RBF units

        # Second subplot: Predicted and Testing Data Wave Approximation
        ax[1].set_title(wave_type + " Wave Approximation with RBF")
        ax[1].plot(test_X, pred_F, label="Predicted Testing Data")
        ax[1].plot(test_X, test_F, label="Testing Data")
        ax[1].legend()

        plt.tight_layout()  # Adjust layout so labels don't overlap
        plt.show()
    




    def plot_2d_weight_space(self, x_range, granularity):
        x_vals_1 = np.linspace(x_range[0], x_range[1], granularity)
        x_vals_2 = np.linspace(x_range[0], x_range[1], granularity)
        X, Y = np.meshgrid(x_vals_1, x_vals_2)
        Z = np.zeros_like(X)
        
        plt.figure(figsize=(8, 6))
        for k in range(self.n_rbf_units):
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


    def plot_rbf_1d_inputs_animated(self, x_range, granularity, pred_F, test_F, wave_type, train_X, train_F):
        """Plot the 1D position of RBF units in the weight space along with their distributions, with animation."""

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))  # Create a figure with two subplots

        # Set titles and labels
        fig.suptitle(f"Radial Basis Function Approximation", fontsize=16)
        ax[0].set_title("1D RBF Unit Distributions in Weight Space")
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("Weighted RBF Output")
        ax[0].grid(True)

        ax[1].set_title(wave_type + " Wave Approximation with RBF")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("Amplitude")
        ax[1].grid(True)

        # Prepare data
        x_vals = np.linspace(x_range[0], x_range[1], granularity)
        line_rbf, = ax[0].plot([], [], lw=2, label="RBF Distribution")  # Line for the RBF unit distribution
        scatter_rbf = ax[0].scatter([], [], color='red')  # Red scatter points for RBF unit centers

        line_pred, = ax[1].plot([], [], label="Predicted Testing Data")  # Line for predicted data
        line_test, = ax[1].plot([], [], label="Testing Data")  # Line for true testing data
        ax[1].legend()

        # Set axis limits
        ax[0].set_xlim(x_range[0], x_range[1])
        ax[0].set_ylim(-2, 2)

        ax[1].set_xlim(0, len(test_F))
        ax[1].set_ylim(min(test_F) - 0.5, max(test_F) + 0.5)

        # Function to initialize the animation
        def init():
            line_rbf.set_data([], [])
            scatter_rbf.set_offsets(np.empty((0, 2)))  # Pass an empty 2D array with shape (0, 2)
            line_pred.set_data([], [])
            line_test.set_data([], [])
            return line_rbf, scatter_rbf, line_pred, line_test


        # Function to animate each frame
        def animate(frame):
            # Dynamically update the weights and centers for each frame
            step_size = 0.1 * (frame + 1)  # Incremental step size from 0.1 to 1.5
            current_mu_list = np.arange(0.1, step_size + 0.1, 0.1)
            current_variance_list = [0.1] * len(current_mu_list)  # Keep variance constant

            num_rbf_units = self.n_rbf_units
            ax[0].text(0.75, 0.95, f'Number of RBF Units: {num_rbf_units}', transform=ax[0].transAxes, 
               fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))


            # Update RBF parameters dynamically
            self.mu = np.array(current_mu_list)
            self.variance = np.array(current_variance_list)
            print("self.w", self.w.shape)
            #self.w = np.random.rand(len(current_mu_list), 1)  # Randomize weights for visualization
            self.batch_supervised_training(train_X, train_F)

            print("self.w", self.w.shape)
        
            # Update RBF plot in the first subplot
            y_vals_rbf = []
            for x in x_vals:
                x_vec = np.array([x])
                rbf_value = 0
                for i in range(len(current_mu_list)):
                    rbf_value += self.w[i] * self.phi(i, x_vec)
                y_vals_rbf.append(rbf_value)

            line_rbf.set_data(x_vals, y_vals_rbf)
            scatter_rbf.set_offsets(np.c_[self.mu, self.w.flatten()])  # Update scatter points for RBF units

            # Update the predicted vs. testing wave in the second subplot
            new_pred_F = self.forward(np.linspace(x_range[0], x_range[1], len(test_F)).reshape(-1, 1))
            line_pred.set_data(np.arange(len(new_pred_F)), new_pred_F)
            line_test.set_data(np.arange(len(test_F)), test_F)

            return line_rbf, scatter_rbf, line_pred, line_test

        num_frames = 15  # Number of animation frames (step sizes from 0.1 to 1.5)
        ani = FuncAnimation(fig, animate, frames=num_frames, init_func=init, blit=True, repeat=True)

        plt.tight_layout()
        plt.show()




    



