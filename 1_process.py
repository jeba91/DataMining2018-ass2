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

#Load the dataset from CSV
try:
    data_all = pd.read_csv("training_set_VU_DM_2014.csv", index_col=0, header=0,  delimiter=",")
except IOError as e:
    print('File not found!')
    raise e

cols_names = data_all.columns

print(cols_names)

data_all = data_all.drop(['visitor_hist_adr_usd', 'visitor_hist_starrating', 'srch_query_affinity_score', 'prop_starrating'])

comp_names = [['comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff'], ['comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff'],
              ['comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff'], ['comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff'],
              ['comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff'], ['comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff'],
              ['comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff'], ['comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff']]

print(comp_names)

total = data_all[comp_names[0][0]].where((data_all[comp_names[0][0]] == 4 & data_all[comp_names[0][1]] == 0),0)

for comp in comp_names:
    summing = data_all[comp[0]].where((data_all[comp[0]] == -1 & data_all[comp[1]] == 0),0)
    total = pd.concat([total,summing],axis=1)

total = total.sum(axis=1)

print(total)
print(total.loc[total < 0].shape)
print(data_all.shape)


# #save data to pickle file
# data_all.to_pickle('preprocessed.pkl')
