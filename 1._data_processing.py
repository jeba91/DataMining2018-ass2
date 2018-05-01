import pandas as pd
import csv

#Load the dataset from CSV
try:
    data_all = pd.read_csv("training_set_VU_DM_2014.csv", index_col=0, header=0,  delimiter=",")
except IOError as e:
    print('File not found!')
    raise e
