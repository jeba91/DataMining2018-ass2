import numpy as np
import random
import pandas as pd
import pyltr

from sklearn.externals import joblib

#Read in preprocessed data
data_train = pd.read_pickle('train.pkl')
data_test = pd.read_pickle('test.pkl')

y_values = data_train['score'].values
data_train.drop(['score'], axis = 1, inplace = True)

columns = data_train.columns
x_values = data_train.values
x_ids = data_train.index.values

print(columns)

metric = pyltr.metrics.NDCG(k=38)

# Only needed if you want to perform validation (early stopping & trimming)
monitor = pyltr.models.monitors.ValidationMonitor(
    x_values, y_values, x_ids, metric=metric, stop_after=250)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=500,
    learning_rate=0.1,
    max_features=0.5,
    query_subsample=0.3,
    max_leaf_nodes=30,
    min_samples_leaf=64,
    verbose=1)

model.fit(x_values, y_values, x_ids, monitor=monitor)
# hallo
joblib.dump(model, 'LambdaMART4.pkl')

y_test = data_test['score'].values
data_test.drop(['score'], axis = 1, inplace = True)
test_data = data_test.values

metric = pyltr.metrics.NDCG(k=38)

query_id = np.asarray(data_test.index.values)
predictions = np.asarray(model.predict(test_data))

print('Random ranking:', metric.calc_mean_random(query_id, y_test))
print('Our model:', metric.calc_mean(query_id, y_test, predictions))
