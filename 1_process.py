import numpy as np
import pandas as pd

#Load the dataset from CSV
try:
    data_all = pd.read_csv("training_set_VU_DM_2014.csv", index_col=0, header=0,  delimiter=",")
except IOError as e:
    print('File not found!')
    raise e

#get column names
cols_names = data_all.columns

#competitor info
comp_names = [['comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff'], ['comp2_rate', 'comp2_inv', 'comp2_rate_percent_diff'],
              ['comp3_rate', 'comp3_inv', 'comp3_rate_percent_diff'], ['comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff'],
              ['comp5_rate', 'comp5_inv', 'comp5_rate_percent_diff'], ['comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff'],
              ['comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff'], ['comp8_rate', 'comp8_inv', 'comp8_rate_percent_diff']]

#CHEAPER AND EXPENSIVE NEW DATA
cheaper = data_all[comp_names[0][0]]
expensive = data_all[comp_names[0][0]]
cheapest = data_all[comp_names[0][2]]
cheaper.iloc[:] = 0
expensive.iloc[:] = 0
cheapest.iloc[:] = 0
cheap_check = cheapest.values
for comp in comp_names:
    conditions1 = [(data_all[comp[0]].values == -1) & (data_all[comp[1]].values == 0)]
    conditions2 = [(data_all[comp[0]].values == 1) & (data_all[comp[1]].values == 0)]
    choices1 = [data_all[comp[0]].values]
    choices2 = [data_all[comp[2]].values]
    cheap = np.select(conditions1, choices1, default=0)
    expen = np.select(conditions2, choices1, default=0)
    max_cheap = np.select(conditions1, choices2, default=0)
    cheaper = cheaper + cheap
    expensive = expensive + expen
    cheap_check = np.maximum(cheap_check, max_cheap)

cheapest = cheapest + cheap_check

data_all['cheaper_comps'] = cheaper
data_all['expensive_comps'] = expensive
data_all['cheapest_comp'] = cheapest

#DROP COMP VARIABLES
for comp in comp_names:
    print(comp)
    data_all.drop(comp, axis = 1, inplace = True)

#DROP REDUNDANT VARIABLE
data_all.drop(['prop_starrating'])

#DROP 4 LARGE NAN VARIABLES
data_all.drop(['visitor_hist_adr_usd', 'visitor_hist_starrating', 'srch_query_affinity_score', 'gross_bookings_usd'], axis = 1, inplace = True)

#Create family column
conditions = [(data_all['srch_children_count'].values == 0) & (data_all['srch_adults_count'].values == 1),
              (data_all['srch_children_count'].values == 0) & (data_all['srch_adults_count'].values == 2),
              (data_all['srch_children_count'].values == 0) & (data_all['srch_adults_count'].values > 2),
              (data_all['srch_children_count'].values > 0)  & (data_all['srch_adults_count'].values == 1),
              (data_all['srch_children_count'].values > 0)  & (data_all['srch_adults_count'].values > 1)]

choices = [1,2,3,4,5]
family_cat = np.select(conditions, choices, default=0)

data_all['family_cat'] = family_cat

#Create score column
conditions = [(data_all['click_bool'].values == 1) & (data_all['booking_bool'].values == 1),
              (data_all['click_bool'].values == 1) & (data_all['booking_bool'].values == 0)]
choices = [5,1]
score = np.select(conditions, choices, default=0)

data_all['score'] = score

#Drop score and other connecting variables
data_all.drop(['click_bool', 'booking_bool', 'date_time', 'random_bool', 'position'], axis = 1, inplace = True)

#save data to pickle file
data_all.to_pickle('preprocessed1.pkl')
