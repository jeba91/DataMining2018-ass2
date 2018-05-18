import numpy as np
import pandas as pd
import random

from sklearn.ensemble import ExtraTreesClassifier
from NDCG_score import NDCGScore

#Read in preprocessed data
data_all = pd.read_pickle('preprocessed3.pkl')

indexes = np.unique(data_all.index.values)
random.shuffle(indexes)

split = int(round(0.2*len(indexes)))
split2 = int(round(0.6*len(indexes)))

index_all = indexes[split:split2]
index_test = indexes[:split]

data_test =  data_all.loc[index_test]
data_all = data_all.loc[index_all]

y_values = data_all['score'].values
data_all.drop(['score'], axis = 1, inplace = True)
columns = data_all.columns
x_values = data_all.values
x_ids = data_all.index.values

clf = ExtraTreesClassifier(n_estimators=1000, max_depth=None, min_samples_split=3, random_state=0, verbose =1)
clf = clf.fit(x_values, y_values)