import pandas as pd
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

#Read in preprocessed data
data_all = pd.read_pickle('preprocessed3.pkl')

data_all.drop(['site_id','prop_country_id','prop_id'], axis = 1, inplace = True)
data_all.drop(['visitor_location_country_id','srch_destination_id'], axis = 1, inplace = True)
data_all.drop(['cheaper_comps','cheapest_comp','expensive_comps','PCA_1','PCA_2'], axis = 1, inplace = True)
data_all.drop(['srch_adults_count','srch_children_count','orig_destination_distance'], axis = 1, inplace = True)

print(data_all.columns)

data_all['prop_hist_price'] = data_all['prop_log_historical_price'].apply(lambda x: 2**x)
data_all['prop_hist_price'].loc[data_all['prop_hist_price'] == 1] = data_all['price_usd'].loc[data_all['prop_hist_price'] == 1]
data_all['diff_price'] = data_all['prop_hist_price'] - data_all['price_usd']
data_all.drop(['prop_hist_price'], axis = 1, inplace = True)

data_all = data_all.sort_index()
data_all['price_rank'] = data_all.groupby(data_all.index)['price_usd'].rank(ascending=False)


indexes = np.unique(data_all.index.values)
random.shuffle(indexes)

split = round(0.2*len(indexes))

index_test = indexes[:split]
index_train = indexes[split:]

data_test  = data_all.loc[index_test]
data_train = data_all.loc[index_train]


#save data to pickle file
data_train.to_pickle('train.pkl')
data_test.to_pickle('test.pkl')
