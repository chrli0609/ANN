import numpy as np

from som import *
from gen_data_func import *
from gen_data_func import *

LEARNING_RATE = 0.2
NUM_EPOCHS = 20
WRAP_AROUND = False

party_mapping = {
    0: 'no party',
    1: 'm',   # Moderates
    2: 'fp',  # Folkpartiet
    3: 's',   # Social Democrats
    4: 'v',   # Left Party
    5: 'mp',  # Green Party
    6: 'kd',  # Christian Democrats
    7: 'c'    # Center Party
}


ファイル道 = "../data/votes.dat"

#Read votes.dat
練習資料 = 動物を読みます(ファイル道, 349, 31)


#Read mpsex.dat
sex = np.loadtxt("../data/mpsex.dat",comments='%')

#Read mpparty.dat
parties = np.loadtxt("../data/mpparty.dat", comments='%')

district = np.loadtxt("../data/mpdistrict.dat",comments='%')

#votes = np.loadtxt("../data/votes.dat")

party_labels = np.array([party_mapping[party] for party in parties])


input_dim = 31
node_rows = 10
node_cols = 10
lattice_dim = 2  # We're using a 2D lattice for the SOM

# Initialize the SOM
som = SOM(input_dim=input_dim, node_rows=node_rows, node_cols=node_cols, lattice_dim=lattice_dim)

som.training(X_data=練習資料, LR=LEARNING_RATE, num_epochs=NUM_EPOCHS, wrap_around=WRAP_AROUND)


# Get BMU for each MP
bmu_indices = [som.get_min_index(練習資料[i]) for i in range(練習資料.shape[0])]

# Visualize BMUs with party, gender, and district labels
def plot_bmu_grid(bmu_indices, attribute_labels, title):
    plt.figure(figsize=(10, 10))
    grid = np.zeros((node_rows, node_cols), dtype=object)
    for i, (x, y) in enumerate(bmu_indices):
        grid[x, y] = attribute_labels[i]

    # Create a unique color mapping for each party
    unique_labels = np.unique(attribute_labels)
    custom_colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'pink', 'brown']
    label_colors = {label: custom_colors[i] for i, label in enumerate(unique_labels)}

    # Plot the grid with labels
    plt.imshow([[label_colors.get(grid[i, j], 0) for j in range(grid.shape[1])] for i in range(grid.shape[0])], cmap='tab20b')
    plt.title(title)
    
    # Create colorbar with party names
    cbar = plt.colorbar(ticks=range(len(unique_labels)))
    cbar.ax.set_yticklabels(unique_labels)
    
    plt.show()

# Plot by party
plot_bmu_grid(bmu_indices, party_labels, "MPs Mapped by Party")

# Plot by gender
plot_bmu_grid(bmu_indices, sex, "MPs Mapped by Gender")

# Plot by district
plot_bmu_grid(bmu_indices, district, "MPs Mapped by District")

