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
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.linear_model import LinearRegression


#Load the dataset from CSV
try:
    data_all = pd.read_csv("test.csv", index_col=0, header=0,  delimiter=",")
except IOError as e:
    print('File not found!')
    raise e

#get column names
cols_names = data_all.columns

#competitor info
comp_names = [['comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff'], ['comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff'],
              ['comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff'], ['comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff'],
              ['comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff'], ['comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff'],
              ['comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff'], ['comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff']]

#CHEAPER AND EXPENSIVE NEW DATA
cheaper = data_all[comp_names[0][0]]
expensive = data_all[comp_names[0][0]]
cheapest = data_all[comp_names[0][2]]
cheaper.iloc[:] = 0
expensive.iloc[:] = 0
cheapest.iloc[:] = 0
cheap_check = cheapest.values
for comp in comp_names:
    conditions1 = [(data_all[comp[0]].values == -1) & (data_all[comp[1]].values == 0)]
    conditions2 = [(data_all[comp[0]].values == 1) & (data_all[comp[1]].values == 0)]
    choices1 = [data_all[comp[0]].values]
    choices2 = [data_all[comp[2]].values]
    cheap = np.select(conditions1, choices1, default=0)
    expen = np.select(conditions2, choices1, default=0)
    max_cheap = np.select(conditions1, choices2, default=0)
    cheaper = cheaper + cheap
    expensive = expensive + expen
    cheap_check = np.maximum(cheap_check, max_cheap)

cheapest = cheapest + cheap_check

data_all['cheaper_comps'] = cheaper
data_all['expensive_comps'] = expensive
data_all['cheapest_comp'] = cheapest

#DROP COMP VARIABLES
for comp in comp_names:
    print(comp)
    data_all.drop(comp, axis = 1, inplace = True)

#DROP REDUNDANT VARIABLE
data_all.drop(['prop_starrating'])

#DROP 4 LARGE NAN VARIABLES
data_all.drop(['visitor_hist_adr_usd', 'visitor_hist_starrating', 'srch_query_affinity_score'], axis = 1, inplace = True)
data_all.drop(['date_time', 'random_bool'], axis = 1, inplace = True)

#Create family column
conditions = [(data_all['srch_children_count'].values == 0) & (data_all['srch_adults_count'].values == 1),
              (data_all['srch_children_count'].values == 0) & (data_all['srch_adults_count'].values == 2),
              (data_all['srch_children_count'].values == 0) & (data_all['srch_adults_count'].values > 2),
              (data_all['srch_children_count'].values > 0)  & (data_all['srch_adults_count'].values == 1),
              (data_all['srch_children_count'].values > 0)  & (data_all['srch_adults_count'].values > 1)]

choices = [1,2,3,4,5]
family_cat = np.select(conditions, choices, default=0)

data_all['family_cat'] = family_cat

x_train = data_all['prop_location_score1'].loc[data_all['prop_location_score2'].notnull()]
y_train = data_all['prop_location_score2'].loc[data_all['prop_location_score2'].notnull()]
x_test = data_all['prop_location_score1'].loc[data_all['prop_location_score2'].isnull()]

logreg = LinearRegression()
logreg.fit(x_train.values.reshape(-1,1), y_train)
y_pred = logreg.predict(x_test.values.reshape(-1,1))

data_all['prop_location_score2'].loc[data_all['prop_location_score2'].isnull()] = y_pred

mean_dist = data_all['orig_destination_distance'].loc[data_all['orig_destination_distance'].notnull()].mean()
data_all['orig_destination_distance'].fillna(mean_dist,inplace=True)

mean_prop_rev = data_all['prop_review_score'].loc[data_all['prop_review_score'].notnull()].mean()
data_all['prop_review_score'].fillna(mean_prop_rev,inplace=True)

data_all['price_usd'].loc[data_all['price_usd'] > 10000] = 100000

data_all['price_usd'] = (data_all['price_usd']-data_all['price_usd'].min())/(data_all['price_usd'].max()-data_all['price_usd'].min())
# data_all['prop_id'] = (data_all['prop_id']-data_all['prop_id'].min())/(data_all['prop_id'].max()-data_all['prop_id'].min())
data_all['srch_destination_id'] = (data_all['srch_destination_id']-data_all['srch_destination_id'].min())/(data_all['srch_destination_id'].max()-data_all['srch_destination_id'].min())
data_all['orig_destination_distance'] = (data_all['orig_destination_distance']-data_all['orig_destination_distance'].min())/(data_all['orig_destination_distance'].max()-data_all['orig_destination_distance'].min())

data_PCA = data_all.copy(deep=True)
columns = data_PCA.columns
columns = columns.append(pd.Index(['PCA1', 'PCA2']))

pca_values = data_PCA.values

pca = PCA().fit(pca_values)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.savefig('visualize/PCA2.png')

pca = PCA(n_components=2)
features = pca.fit_transform(pca_values)
features_analyse = pca.fit(pca_values)

features = features.transpose()

data_PCA['PCA_1'] = features[0]
data_PCA['PCA_2'] = features[1]


#save data to pickle file
data_PCA.to_pickle('test.pkl')
