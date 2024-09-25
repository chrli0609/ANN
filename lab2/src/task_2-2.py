from som import *
from gen_data_func import *
from gen_data_func import *

LEARNING_RATE = 0.2
NUM_EPOCHS = 20
WRAP_AROUND = True

city_names = ["City 1", "City 2", "City 3", "City 4", "City 5", "City 6", "City 7", "City 8", "City 9", "City 10"]


練習資料 = [
    [0.4000, 0.4439],
    [0.2439, 0.1463],
    [0.1707, 0.2293],
    [0.2293, 0.7610],
    [0.5171, 0.9414],
    [0.8732, 0.6536],
    [0.6878, 0.5219],
    [0.8488, 0.3609],
    [0.6683, 0.2536],
    [0.6195, 0.2634]
]



model = SOM(input_dim=len(練習資料[0]), node_rows=1, node_cols=10, lattice_dim=1)


model.plot_tsp_solution(np.array(練習資料), namelist=city_names)


model.training(np.array(練習資料), LEARNING_RATE, NUM_EPOCHS, WRAP_AROUND)


model.plot_tsp_solution(np.array(練習資料), namelist=city_names)
