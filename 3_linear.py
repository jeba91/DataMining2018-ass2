import numpy as np
import pandas as pd
import random

from sklearn.linear_model import LinearRegression
from imblearn.under_sampling import RandomUnderSampler
from NDCG_score import NDCGScore

#Read in preprocessed data
data_all = pd.read_pickle('preprocessed3.pkl')

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
x_train, y_train, idx_resampled = rus.fit_sample(x_values, y_values)

# x_resampled, y_resampled = x_values, y_values


estimator = LinearRegression()
estimator.fit(x_train,y_train)
print(estimator)



total = []

for i in index_test:
    data_check = data_test.loc[i].copy(deep=True)
    y_test = data_check['score'].values
    data_check.drop(['score'], axis = 1, inplace = True)
    x_test = data_check.values
    scores = estimator.predict(x_test)
    predictions = [x for _,x in sorted(zip(scores,y_test), reverse=True)]
    # max_predict = sorted(y_test, reverse=True)
    ndcg_score = ndcg_at_k(predictions,38,method=1)
    total.append(ndcg_score)
    print(ndcg_score)

print("NDCG is ", np.mean(total))
