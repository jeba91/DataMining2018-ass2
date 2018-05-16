import copy
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import pandas
import pandas as pd
import datetime as dt
import seaborn as sns
from pandas import DataFrame
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from datetime import datetime
from tabulate import tabulate
import numpy

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVC
import pandas as pd
from sklearn import preprocessing
from imblearn.under_sampling import RandomUnderSampler

#Read in preprocessed data
data_all = pd.read_pickle('preprocessed2.pkl')

y_values = data_all['score'].values
data_all.drop(['score'], axis = 1, inplace = True)

columns = data_all.columns
columns = columns.append(pd.Index(['PCA1', 'PCA2', 'PCA3']))
print(columns)

data_all['price_usd'].loc[data_all['price_usd'] > 10000] = 100000

data_all['price_usd'] = (data_all['price_usd']-data_all['price_usd'].min())/(data_all['price_usd'].max()-data_all['price_usd'].min())
data_all['prop_id'] = (data_all['prop_id']-data_all['prop_id'].min())/(data_all['prop_id'].max()-data_all['prop_id'].min())
data_all['srch_destination_id'] = (data_all['srch_destination_id']-data_all['srch_destination_id'].min())/(data_all['srch_destination_id'].max()-data_all['srch_destination_id'].min())
data_all['orig_destination_distance'] = (data_all['orig_destination_distance']-data_all['orig_destination_distance'].min())/(data_all['orig_destination_distance'].max()-data_all['orig_destination_distance'].min())



x_values = data_all.values

# Apply the random under-sampling
rus = RandomUnderSampler(return_indices=True)
x_resampled, y_resampled, idx_resampled = rus.fit_sample(x_values, y_values)

pca = PCA().fit(x_resampled)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.savefig('visualize/PCA2.png')

pca = PCA(n_components=3)
features = pca.fit_transform(x_resampled)
features_analyse = pca.fit(x_resampled)

# print(pca.explained_variance_)
x_train = np.concatenate((x_resampled, features), axis=1)
y_train = y_resampled




# print ("hallo")
# #RandomForestEstimator
# estimator = SVC(kernel='linear', C=1, max_iter=5000)
# print(estimator)
# selector = RFECV(estimator, step=1, cv=3, scoring='accuracy', verbose=1)
# selector = selector.fit(x_train, y_train)
#
# from sklearn.externals import joblib
# joblib.dump(selector, 'selector.pkl')
#
# selector = joblib.load('selector.pkl')
# print('Optimal number of features :', selector.n_features_)
# print('Best features :', selector.support_)
#
# indexes = [i for i, x in enumerate(selector.support_) if x]
#
# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (min value of MSE)")
# plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
# plt.show()




#UNCOMMENT FOR TREE
# Build a forest and compute the feature importances
forest = ExtraTreesRegressor(n_estimators=250, random_state=0)
forest.fit(x_train, y_train)

importances = forest.feature_importances_
print(importances)
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

total = 0

for f in range(x_train.shape[1]):
    total = total + importances[indices[f]]
    print(f + 1, columns[indices[f]-1], indices[f], importances[indices[f]], total)

plot_names = [columns[indices[f]-1] for f in range(len(indices))]
print(plot_names)

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), plot_names)
plt.xticks(rotation='vertical')
plt.xlim([-1, x_train.shape[1]])
plt.tight_layout()
plt.savefig('visualize/tree_importance.png')
