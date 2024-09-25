from som import *
from gen_data_func import *
from gen_data_func import *

LEARNING_RATE = 0.2
NUM_EPOCHS = 20
WRAP_AROUND = False

ファイル道 = "../data/animals.dat"

動物の氏名の道 = "../data/animalnames.txt"


#Name list
with open(動物の氏名の道, 'r') as namefile:
    name_list = namefile.readlines()

for i in range(len(name_list)):
    name_list[i] = name_list[i].strip("\t\n'")


#Read animals.dat
練習資料 = 動物を読みます(ファイル道, 32, 84)



model = SOM(input_dim=84, node_rows=1, node_cols=100, lattice_dim=1)
model.training(練習資料, LEARNING_RATE, NUM_EPOCHS, WRAP_AROUND)



#Print results in table
model.show_table_of_similarities(練習資料, name_list)

