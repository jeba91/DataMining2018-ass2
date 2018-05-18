import pandas as pd

from imblearn.under_sampling import RandomUnderSampler

#Read in preprocessed data
data_all = pd.read_pickle('preprocessed3.pkl')

y_values = data_all['score'].values
data_all.drop(['score'], axis = 1, inplace = True)
columns = data_all.columns
x_values = data_all.values


# Apply the random under-sampling
rus = RandomUnderSampler(return_indices=True)
x_train, y_train, idx_resampled = rus.fit_sample(x_values, y_values)

# x_resampled, y_resampled = x_values, y_values


# print ("hallo")
# #RandomForestEstimator
# estimator = LinearRegression()
# print(estimator)
# selector = RFECV(estimator, step=1, cv=3, scoring='neg_mean_absolute_error', verbose=1)
# selector = selector.fit(x_train, y_train)
#
# from sklearn.externals import joblib
# joblib.dump(selector, 'selector.pkl')
#
# selector = joblib.load('selector.pkl')
# print('Optimal number of features :', selector.n_features_)
# print('Best features :', selector.support_)
#
# indexes = [i for i, x in enumerate(selector.support_) if x]
#
# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (min value of MSE)")
# plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
# plt.show()




# #UNCOMMENT FOR TREE
# # Build a forest and compute the feature importances
# forest = ExtraTreesRegressor(n_estimators=250, random_state=0)
# forest.fit(x_train, y_train)
#
# importances = forest.feature_importances_
# print(importances)
# std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
# indices = np.argsort(importances)[::-1]
#
# # Print the feature ranking
# print("Feature ranking:")
#
# total = 0
#
# for f in range(x_train.shape[1]):
#     total = total + importances[indices[f]]
#     print(f + 1, columns[indices[f]-1], indices[f], importances[indices[f]], total)
#
# plot_names = [columns[indices[f]-1] for f in range(len(indices))]
# print(plot_names)
#
# # Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(x_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
# plt.xticks(range(x_train.shape[1]), plot_names)
# plt.xticks(rotation='vertical')
# plt.xlim([-1, x_train.shape[1]])
# plt.tight_layout()
# plt.savefig('visualize/tree_importance.png')
