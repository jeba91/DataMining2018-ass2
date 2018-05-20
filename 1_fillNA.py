import pandas as pd

from sklearn.linear_model import LinearRegression

#Read in preprocessed data
data_all = pd.read_pickle('preprocessed1.pkl')

x_train = data_all['prop_location_score1'].loc[data_all['prop_location_score2'].notnull()]
y_train = data_all['prop_location_score2'].loc[data_all['prop_location_score2'].notnull()]
x_test = data_all['prop_location_score1'].loc[data_all['prop_location_score2'].isnull()]

logreg = LinearRegression()
logreg.fit(x_train.values.reshape(-1,1), y_train)
y_pred = logreg.predict(x_test.values.reshape(-1,1))

data_all['prop_location_score2'].loc[data_all['prop_location_score2'].isnull()] = y_pred

mean_dist = data_all['orig_destination_distance'].loc[data_all['orig_destination_distance'].notnull()].mean()
data_all['orig_destination_distance'].fillna(mean_dist,inplace=True)

mean_prop_rev = data_all['prop_review_score'].loc[data_all['prop_review_score'].notnull()].mean()
data_all['prop_review_score'].fillna(mean_prop_rev,inplace=True)

data_all['price_usd'].loc[data_all['price_usd'] > 10000] = 100000

data_all['price_usd'] = (data_all['price_usd']-data_all['price_usd'].min())/(data_all['price_usd'].max()-data_all['price_usd'].min())
data_all['srch_destination_id'] = (data_all['srch_destination_id']-data_all['srch_destination_id'].min())/(data_all['srch_destination_id'].max()-data_all['srch_destination_id'].min())
data_all['orig_destination_distance'] = (data_all['orig_destination_distance']-data_all['orig_destination_distance'].min())/(data_all['orig_destination_distance'].max()-data_all['orig_destination_distance'].min())

#save data to pickle file
data_all.to_pickle('preprocessed2.pkl')
