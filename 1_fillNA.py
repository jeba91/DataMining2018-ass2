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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import mean_absolute_error

#Read in preprocessed data
data_all = pd.read_pickle('preprocessed1.pkl')

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
data_all['prop_id'] = (data_all['prop_id']-data_all['prop_id'].min())/(data_all['prop_id'].max()-data_all['prop_id'].min())
data_all['srch_destination_id'] = (data_all['srch_destination_id']-data_all['srch_destination_id'].min())/(data_all['srch_destination_id'].max()-data_all['srch_destination_id'].min())
data_all['orig_destination_distance'] = (data_all['orig_destination_distance']-data_all['orig_destination_distance'].min())/(data_all['orig_destination_distance'].max()-data_all['orig_destination_distance'].min())

#save data to pickle file
data_all.to_pickle('preprocessed2.pkl')
