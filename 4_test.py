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
from sklearn.externals import joblib

import pyltr

# #Read in preprocessed data
# data_all = pd.read_pickle('test_kaggle.pkl')

#Read in preprocessed data
data_all = pd.read_pickle('preprocessed3.pkl')


import time
tic = time.clock()

model = joblib.load('lambdaMART3.pkl')

test_data = data_all.copy(deep=True)
y_test = data_all['score'].values
data_all.drop(['score'], axis = 1, inplace = True)

test_data.drop(['site_id','prop_country_id','prop_id'], axis = 1, inplace = True)
test_data.drop(['visitor_location_country_id','srch_destination_id'], axis = 1, inplace = True)


test_data = test_data.values


metric = pyltr.metrics.NDCG(k=38)

query_id = np.asarray(data_all.index.values)
prop_id = np.asarray(data_all['prop_id'])
predictions = np.asarray(model.predict(test_data))

print 'Random ranking:', metric.calc_mean_random(query_id, y_test)
print 'Our model:', metric.calc_mean(query_id, y_test, predictions)

#
# data = np.concatenate((query_id.reshape((-1, 1)),prop_id.reshape((-1, 1))),axis=1 )
# data = np.concatenate((data,predictions.reshape((-1, 1))),axis=1 )
#
# data_predic = pd.DataFrame(data,columns = ['SearchId','PropertyId','predictions'])
#
# data_predic = data_predic.groupby(['SearchId']).apply(lambda x: x.sort_values(["predictions"])).reset_index(drop=True)
# # , ascending = False
# data_predic.SearchId = data_predic.SearchId.astype(int)
# data_predic.PropertyId = data_predic.PropertyId.astype(int)
#
# print(data_predic)
#
# data_predic.drop(['predictions'], axis = 1, inplace = True)
#
# data_predic.to_csv('predictions.csv', encoding='utf-8', index=False)
#
# toc = time.clock()
# print(toc - tic)
