import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

#Read in preprocessed data
data_all = pd.read_pickle('preprocessed2.pkl')

data_PCA = data_all.copy(deep=True)
data_PCA.drop(['score'], axis = 1, inplace = True)

columns = data_PCA.columns
columns = columns.append(pd.Index(['PCA1', 'PCA2']))

pca_values = data_PCA.values

pca = PCA().fit(pca_values)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.savefig('visualize/PCA2.png')

pca = PCA(n_components=2)
features = pca.fit_transform(pca_values)
features_analyse = pca.fit(pca_values)

features = features.transpose()

data_PCA['PCA_1'] = features[0]
data_PCA['PCA_2'] = features[1]
data_PCA['score'] = data_all['score'].values

#save data to pickle file
data_PCA.to_pickle('preprocessed3.pkl')
