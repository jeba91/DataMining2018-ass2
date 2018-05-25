import pandas as pd

from sklearn.linear_model import LinearRegression

#Read in preprocessed data
data_all = pd.read_pickle('preprocessed3.pkl')

data_all.drop(['site_id','prop_country_id','prop_id'], axis = 1, inplace = True)
data_all.drop(['visitor_location_country_id','srch_destination_id'], axis = 1, inplace = True)
data_all.drop(['cheaper_comps','cheapest_comp','expensive_comps','PCA_1','PCA_2'], axis = 1, inplace = True)
data_all.drop(['srch_adults_count','srch_children_count','orig_destination_distance'], axis = 1, inplace = True)

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
