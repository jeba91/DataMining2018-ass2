import pandas as pd
import csv

#Load the dataset from CSV
try:
    data_all = pd.read_csv("training_set_VU_DM_2014.csv", index_col=0, header=0,  delimiter=",")
except IOError as e:
    print('File not found!')
    raise e

#Scatterplot of single feature.
def scatter(data, col):
    plt.scatter(range(len(data[col])), data[col], s=1)
    plt.xlabel('timestamp')
    plt.ylabel('value')
    plt.title(col)
    plt.show()
    return


#Plot distribution of values.
def distribution(data, col):
    datapoints = data['proc_gyro:3d:mean_z'].value_counts()
    plt.scatter(datapoints.index.values, datapoints, s=1)
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.title(col + ' distribution')
    plt.show()
    return


#Scatterplot single feature colored by label.
def scatter_color(data, col, labels):
    colors = ["r", "b", "g", "y", "m"]
    labels = data[labels].unique()
    for i, j in zip(labels, colors):
        plt.scatter(range(len(data.groupby('labels')[col].get_group(i))),
        data.groupby('labels')[col].get_group(i), color=j, s=1, label=i)
    plt.xlabel('timestamp')
    plt.ylabel('value')
    plt.title('col')
    plt.legend(bbox_to_anchor=(1.2, 1.1))
    plt.show()
    return

scatter(data, col)
