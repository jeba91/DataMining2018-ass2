import numpy as np
import random
import pandas as pd
import pyltr

from sklearn.externals import joblib

#Read in preprocessed data
data_all = pd.read_pickle('preprocessed3.pkl')

print(data_all.columns)

data_all.drop(['site_id','prop_country_id','prop_id'], axis = 1, inplace = True)
data_all.drop(['visitor_location_country_id','srch_destination_id'], axis = 1, inplace = True)
data_all.drop(['cheaper_comps','cheapest_comp','expensive_comps','PCA_1','PCA_2'], axis = 1, inplace = True)
data_all.drop(['srch_adults_count','srch_children_count','orig_destination_distance'], axis = 1, inplace = True)

indexes = np.unique(data_all.index.values)
random.shuffle(indexes)

split = int(round(0.2*len(indexes)))

index_all = indexes[split:]
index_test = indexes[:split]

data_test =  data_all.loc[index_test]
data_all = data_all.loc[index_all]

y_values = data_all['score'].values
data_all.drop(['score'], axis = 1, inplace = True)


columns = data_all.columns
x_values = data_all.values
x_ids = data_all.index.values

print(columns)

metric = pyltr.metrics.NDCG(k=38)

# Only needed if you want to perform validation (early stopping & trimming)
monitor = pyltr.models.monitors.ValidationMonitor(
    x_values, y_values, x_ids, metric=metric, stop_after=250)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=500,
    learning_rate=0.04,
    max_features=0.5,
    query_subsample=0.3,
    max_leaf_nodes=30,
    min_samples_leaf=64,
    verbose=1,)

model.fit(x_values, y_values, x_ids, monitor=monitor)

joblib.dump(model, 'LambdaMART4.pkl')

y_test = data_test['score'].values
data_all.drop(['score'], axis = 1, inplace = True)
test_data = data_test.values

metric = pyltr.metrics.NDCG(k=38)

query_id = np.asarray(data_test.index.values)
predictions = np.asarray(model.predict(test_data))

print 'Random ranking:', metric.calc_mean_random(query_id, y_test)
print 'Our model:', metric.calc_mean(query_id, y_test, predictions)
