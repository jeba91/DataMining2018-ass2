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
from sklearn.svm import SVC
import pandas as pd
from sklearn import preprocessing
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import random




import numpy as np
import pandas as pd
import random

from sklearn.ensemble import AdaBoostRegressor
from sklearn.externals import joblib

from imblearn.under_sampling import RandomUnderSampler

def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

import time
tic = time.clock()

#Read in preprocessed data
data_train = pd.read_pickle('train.pkl')
data_test = pd.read_pickle('test.pkl')

y_values = data_train['score'].values
data_train.drop(['score'], axis = 1, inplace = True)

columns = data_train.columns
x_values = data_train.values

# Apply the random under-sampling
rus = RandomUnderSampler(return_indices=True)
x_train, y_train, idx_resampled = rus.fit_sample(x_values, y_values)
# x_train, y_train = x_values, y_values


ada = AdaBoostRegressor(n_estimators=500)
ada.fit(x_train, y_train)

joblib.dump(ada, 'AdaBoost.pkl')

query_id = np.asarray(data_test.index.values)
y_test = data_test['score'].values
data_test.drop(['score'], axis = 1, inplace = True)
x_test = data_test.values

predictions = ada.predict(x_test)

data = np.concatenate((query_id.reshape((-1, 1)),y_test.reshape((-1, 1))),axis=1 )
data = np.concatenate((data,predictions.reshape((-1, 1))),axis=1 )

data_predic = pd.DataFrame(data, columns = ['SearchId','score','predictions'])
data_predic = data_predic.groupby(['SearchId']).apply(lambda x: x.sort_values(["predictions"], ascending = False)).reset_index(drop=True)

data_new = data_predic.groupby(['SearchId']).apply(lambda x: ndcg_at_k(x['score'].values,38,1))

toc = time.clock()
print("Done in " , toc - tic ,"seconds")
print("NDCG is ", data_new.mean())
