import numpy as np
import random
import pandas as pd
import pyltr

from sklearn.externals import joblib

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


# # Apply the random under-sampling
# rus = RandomUnderSampler(return_indices=True)
# x_train, y_train, idx_resampled = rus.fit_sample(x_values, y_values)

metric = pyltr.metrics.NDCG(k=38)

# Only needed if you want to perform validation (early stopping & trimming)
monitor = pyltr.models.monitors.ValidationMonitor(
    x_values, y_values, x_ids, metric=metric, stop_after=250)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=1000,
    learning_rate=0.05,
    max_features=0.5,
    query_subsample=0.3,
    max_leaf_nodes=7,
    min_samples_leaf=64,
    verbose=1,)

model.fit(x_values, y_values, x_ids, monitor=monitor)

joblib.dump(model, 'LambdaMART2.pkl')