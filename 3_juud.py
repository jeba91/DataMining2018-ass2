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
from sklearn.linear_model import Perceptron
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

y_values = data_all['score'].values
data_all.drop(['score'], axis = 1, inplace = True)

columns = data_all.columns
columns = columns.append(pd.Index(['PCA1', 'PCA2', 'PCA3']))
print(columns)

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
