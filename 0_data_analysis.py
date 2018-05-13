import copy
import os
import numpy as np

import math
import pandas
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.random as random
from pandas import DataFrame
from datetime import datetime
import numpy
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Functions for data analysis and data cleaning/interpolation
class visualize_functions:
    #Scatterplot function
    def scatterplot(self, dataset, variables):
        #for subplots
        fig, axes = plt.subplots(3,3, sharey=True )
        a = 0
        b = 0

        #scatter plot totalbsmtsf/saleprice
        for col in variables:
            if var == 'mood':
                break
            data = pd.concat([dataset['mood'], dataset[var]], axis=1)
            data.plot.scatter(ax=axes[a,b], x=var, y='mood'); axes[a,b].set_title(var)
            a = a+1
            if a == 3:
                b = b+1
                a = 0

        fig.savefig('visualize/scatterplot.png')

    #QQ plot function
    def QQplot(self, dataset, variables):
        for var in variables:
            data_var = dataset[var].values.flatten()
            data_var.sort()
            norm  = random.normal(0,2,len(data_var))
            norm.sort()
            fig = plt.figure(figsize=(12,8),facecolor='1.0')
            plt.plot(norm,data_var,"o")
            z = np.polyfit(norm,data_var, 1)
            p = np.poly1d(z)
            plt.plot(norm,p(norm),"k--", linewidth=2)
            plt.title("Normal Q-Q plot "+var, size=28)
            plt.xlabel("Theoretical quantiles", size=24)
            plt.ylabel("Expreimental quantiles", size=24)
            plt.tick_params(labelsize=16)
            fig.savefig('visualize/'+var+'.png')

    #heatmap function
    def heatmap_corr(self, dataset):
        #UNCOMMENT FOR PLOTTING CORRELATION MATRIX
        corrmat = dataset.corr()
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=.8, square=True);
        plt.xticks(rotation='vertical')
        plt.yticks(rotation='horizontal')
        fig.savefig('visualize/heatmap.png')


#Load the dataset from CSV
try:
    data_all = pd.read_csv("training_set_VU_DM_2014.csv", index_col=0, header=0,  delimiter=",")
except IOError as e:
    print('File not found!')
    raise e

func = visualize_functions()

##HEATMAP AND DESCRIBE
# func.heatmap_corr(data_all)
# data_all.describe().to_csv('describe_data.csv')

#PLOT NAN VALUES
cols_names = data_all.columns
nandat = data_all.isnull().sum()
cols_names = [x for _,x in sorted(zip(nandat,cols_names))]
nandat = nandat.sort_values()
total = data_all.shape[0]
nandat = [(e/total)*100 for e in nandat]
sorted = range(len(cols_names))
plt.bar(sorted, nandat, color="#3F5D7D", align='edge', width=0.6)
plt.ylabel('Percentage NaN values')
plt.xticks(sorted, cols_names, rotation='vertical')
plt.tight_layout()
plt.savefig('visualize/nanvalues.png')
plt.close()

##PLOT BOOKING WINDOW VALUES
# range_book = [[0,2],[3,5],[6,],[12,20],[21,50],[50,500000000000]]
# range_book_str = ['zero to two','three to five','six to eleven','twelve to twenty','twentyone to fifty','fifty plus']
#
# range_book = range(0,100)
# range_book_str = [str(e) for e in range_book]
# all2 = []
#
# for ran in range_book:
#     count = data_all.loc[(data_all['srch_booking_window'] == ran) & (data_all['booking_bool'] == 1)].shape[0]
#     all2.append(count)
#
# print(range_book_str)
#
# plt.bar(range(len(range_book_str)), all2, 0.25, color="blue")
# plt.xticks(range(len(range_book_str)),range_book_str)
# plt.xticks(rotation='vertical')
# plt.savefig('visualize/book_window.png')
# plt.tight_layout()
# plt.show()

##PLOT BOOKED PER STAR
# scores = np.sort(data_all.prop_review_score.unique())
#
# print(data_all['booking_bool'])
#
# all = []
#
# for s in scores:
#     count = data_all.loc[(data_all['prop_review_score'] == s) & (data_all['booking_bool'] == 1)].shape[0]
#     all.append(count)
#
# print(scores)
# print(all)
#
# y = [3, 10, 7, 5, 3, 4.5, 6, 8.1]
# N = len(y)
# x = range(N)
# width = 0.25
# plt.bar(scores, all, width, color="blue")
# plt.savefig('visualize/bookedperstar.png')
# plt.show()

#PLOT SCATTER
# pd.plotting.scatter_matrix(data_all, diagonal='kde')
# plt.show()
# fig.savefig('visualize/scatter.png')
# plt.close(fig)

#MAKE AL BOX, DENSITY AND HISTOGRAMS
#make boxplot, density plot and histogram
# for col in data_all:
#     if col == 'date_time':
#         continue
#     fig, axes = plt.subplots(1,3, figsize=(25,8))
#     print(axes)
#     data_all.boxplot(column=col, ax=axes[0])
#     data_all[col].plot(kind='kde', ax=axes[1])
#     data_all.hist(column=col, ax=axes[2])
#     fig.savefig('visualize/' + col + '.png')
#     plt.close(fig)    # close the figure
