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
from sklearn.linear_model import LogisticRegression
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

data_PCA = data_all.copy(deep=True)
data_PCA.drop(['score'], axis = 1, inplace = True)

columns = data_PCA.columns
columns = columns.append(pd.Index(['PCA1', 'PCA2', 'PCA3']))

pca_values = data_PCA.values

pca = PCA().fit(pca_values)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.savefig('visualize/PCA2.png')

pca = PCA(n_components=3)
features = pca.fit_transform(pca_values)
features_analyse = pca.fit(pca_values)

features = features.transpose()

data_PCA['PCA_1'] = features[0]
data_PCA['PCA_2'] = features[1]
data_PCA['PCA_3'] = features[2]
data_PCA['score'] = data_all['score'].values

#save data to pickle file
data_PCA.to_pickle('preprocessed3.pkl')
