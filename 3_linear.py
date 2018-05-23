import numpy as np
import pandas as pd
import random

from sklearn.linear_model import LinearRegression
from imblearn.under_sampling import RandomUnderSampler

from sklearn.externals import joblib

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
data_all = pd.read_pickle('preprocessed3.pkl')

data_all.drop(['site_id','prop_country_id','prop_id'], axis = 1, inplace = True)
data_all.drop(['visitor_location_country_id','srch_destination_id'], axis = 1, inplace = True)

indexes = np.unique(data_all.index.values)
random.shuffle(indexes)

split = round(0.2*len(indexes))

index_all = indexes[split:]
index_test = indexes[:split]

data_test =  data_all.loc[index_test]
data_all = data_all.loc[index_all]

y_values = data_all['score'].values
data_all.drop(['score'], axis = 1, inplace = True)

columns = data_all.columns
x_values = data_all.values

# Apply the random under-sampling
rus = RandomUnderSampler(return_indices=True)
# x_train, y_train, idx_resampled = rus.fit_sample(x_values, y_values)
x_train, y_train = x_values, y_values

# from sklearn.svm import SVR
# estimator = SVR()

estimator = LinearRegression()
estimator.fit(x_train,y_train)

joblib.dump(estimator, 'linear.pkl')

query_id = np.asarray(data_test.index.values)
y_test = data_test['score'].values
data_test.drop(['score'], axis = 1, inplace = True)
x_test = data_test.values

predictions = estimator.predict(x_test)

data = np.concatenate((query_id.reshape((-1, 1)),y_test.reshape((-1, 1))),axis=1 )
data = np.concatenate((data,predictions.reshape((-1, 1))),axis=1 )

data_predic = pd.DataFrame(data, columns = ['SearchId','score','predictions'])
data_predic = data_predic.groupby(['SearchId']).apply(lambda x: x.sort_values(["predictions"], ascending = False)).reset_index(drop=True)

data_new = data_predic.groupby(['SearchId']).apply(lambda x: ndcg_at_k(x['score'].values,38,1))

toc = time.clock()
print("Done in " , toc - tic ,"seconds")
print("NDCG is ", data_new.mean())