import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from prettytable import PrettyTable



class Node():

    def __init__(self, dim):

        self.weights = []
        for i in range(dim):
            self.weights.append(np.random.rand())
            #self.weights.append(1)

    
    def weight_update(self, lr, h, x_sample):
        self.weights += lr * h * (x_sample - self.weights)

    

    


class SOM():
    def __init__(self, input_dim, node_rows, node_cols, lattice_dim):

        self.input_dim = input_dim
        self.node_cols = node_cols
        self.node_rows = node_rows
        self.lattice_dim = lattice_dim



        #Create lattice structure
        lattice = []
        for i in range(node_rows):
            row_list = []
            for j in range(node_cols):
                row_list.append(Node(input_dim))

            lattice.append(row_list)
        
        self.lattice = np.array(lattice)

    


    def neighborhood_func(self, epoch, bmu_x, bmu_y, wrap_around):


        neighbors = np.zeros((self.node_rows, self.node_cols))
        #print("self.node_cols", self.node_cols, "self.node_rows", self.node_rows)
        for x in range(self.node_rows):
            for y in range(self.node_cols):
                
                # Calculate the toroidal distance
                
                if wrap_around:
                    dx = min(abs(bmu_x - x), self.node_rows - abs(bmu_x - x))
                    dy = min(abs(bmu_y - y), self.node_cols - abs(bmu_y - y))
                    #distance = np.sqrt(dx**2 + dy**2)
                    neighbor_lvl = dx + dy
                else:
                    dx = abs(bmu_x - x)
                    dy = abs(bmu_y - y)

                    neighbor_lvl = dx + dy
                
                neighbors[x, y] = gauss_func(epoch, neighbor_lvl)
        #print(neighbors)

        return neighbors

    
    def get_min_index(self, sample):
        d_vec = np.zeros((self.node_rows, self.node_cols))
        for j in range(self.node_cols):
            for k in range(self.node_rows):
                #print("X_data", X_data.shape)
                #print("self.lattice[k,j].weights)", len(self.lattice[k,j].weights))
                d_vec[k,j] = (np.linalg.norm(sample - (self.lattice[k,j].weights)))
        #print("d_vec,", d_vec)
        min_row, min_col = min_arg(d_vec)
        #print("min_row, min_col", min_row, min_col)

        return min_row, min_col


    def training(self, X_data, LR, num_epochs, wrap_around):
        #X_data is ()
        n_samples, _ = X_data.shape
        for epoch in range(num_epochs):
            for i in range(n_samples):
                #Find BMU
                min_row, min_col = self.get_min_index(X_data[i])

                #Compute Neighborhodd function
                h_lattice = self.neighborhood_func(epoch, min_row, min_col, wrap_around)

                #Update all weights
                self.update_all_weights(LR, h_lattice, X_data[i])
                

        return 

    def update_all_weights(self, lr, h_lattice, sample):

        for i in range(len(self.lattice)):
            for j in range(len(self.lattice[i])):
                self.lattice[i,j].weight_update(lr, h_lattice[i,j], sample)


    def show_table_of_similarities(self, train_data, namelist):
        """
        Generates a topographical map based on the Euclidean distances between neighboring nodes.
        """
        #topo_map = np.zeros((self.node_rows, self.node_cols))

        name_bmu_indices = []
        
        #For each sample
        for i in range(len(namelist)):
            
            """for j in range(len(self.lattice[0])):
                print("train_data[i]", train_data[i])
                print("self.lattice[0][j].weights",self.lattice[0][j].weights)"""

            #Check which is the BMU
            min_row, min_col = self.get_min_index(train_data[i])

            name_bmu_indices.append([min_row, min_col])
        
        


        name_bmu_map = []
        for i in range(len(namelist)):
            name_bmu_map.append([name_bmu_indices[i][1], namelist[i]])

      
        name_bmu_map = sorted(name_bmu_map, key = lambda y_axis: y_axis[0], reverse = False)
        table = PrettyTable()
        
        table.field_names = ["BMU", "Animal"]
        for i in range(len(name_bmu_map)):
            if i+1 < len(name_bmu_map) and name_bmu_map[i][0] != name_bmu_map[i+1][0]:
                divider = True
            else:
                divider = False

            table.add_row([name_bmu_map[i][0], name_bmu_map[i][1]], divider=divider)
            
        print(table)

    
    def plot_tsp_solution(self, cities, namelist=None):
        """
        Plots the cities and the SOM path for the traveling salesman problem.
        
        :param cities: List of 2D coordinates of cities (input data)
        :param namelist: Optional list of names for the cities for labeling
        """
        # Get the BMU indices for the cities
        bmu_indices = []
        for i in range(len(cities)):
            min_row, min_col = self.get_min_index(cities[i])
            bmu_indices.append(min_col)

        # Sort cities according to the BMU indices
        sorted_indices = np.argsort(bmu_indices)
        sorted_cities = np.array(cities)[sorted_indices]

        # Plot the cities and the path
        plt.figure(figsize=(8, 6))
        plt.scatter(cities[:, 0], cities[:, 1], color='red', label='Cities')

        # Draw lines between cities in the TSP order
        for i in range(len(sorted_cities) - 1):
            plt.plot([sorted_cities[i][0], sorted_cities[i + 1][0]],
                     [sorted_cities[i][1], sorted_cities[i + 1][1]], 'k-', lw=1)

        # Connect the last city back to the first to complete the tour
        plt.plot([sorted_cities[-1][0], sorted_cities[0][0]],
                 [sorted_cities[-1][1], sorted_cities[0][1]], 'k-', lw=1)

        # Optionally, add labels for each city
        if namelist is not None:
            for i, name in enumerate(namelist):
                plt.text(cities[i][0], cities[i][1], name, fontsize=12)

        plt.title('Traveling Salesman Problem Solution')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.show()






def min_arg(matrix):

    smallest_value = sys.maxsize
    smallest_row_index = 0
    smallest_col_index = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] < smallest_value:
                smallest_value = matrix[i][j]
                smallest_row_index = i
                smallest_col_index = j
    
    return smallest_row_index, smallest_col_index
    
def gauss_func(epoch, neighbour_lvl):
    #Animals
    """sigma0 = 14
    tao = 125"""
    sigma0 = 4
    tao = 50
    sigma = sigma0*math.exp(-pow(epoch,2)/tao)
    
    h = math.exp(-pow(neighbour_lvl, 2)/(2*pow(sigma,2)))
    
    return h


"""
print(gauss_func(0,0))

import matplotlib.pyplot as plt
x = np.linspace(-10, 10, 20)
y = []
for i in range(len(x)):
    y.append(gauss_func(0, x[i]))
plt.plot(x, y)
plt.show()"""
