import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.random as random

# Functions for data analysis and data cleaning/interpolation
class visualize_functions:
    #Scatterplot function
    def scatterplot(self, dataset, variables):
        #for subplots
        fig, axes = plt.subplots(3,3, sharey=True )
        a = 0
        b = 0

        #scatter plot totalbsmtsf/saleprice
        for col in variables:
            if var == 'mood':
                break
            data = pd.concat([dataset['mood'], dataset[var]], axis=1)
            data.plot.scatter(ax=axes[a,b], x=var, y='mood'); axes[a,b].set_title(var)
            a = a+1
            if a == 3:
                b = b+1
                a = 0

        fig.savefig('visualize/scatterplot.png')

    #QQ plot function
    def QQplot(self, dataset, variables):
        for var in variables:
            data_var = dataset[var].values.flatten()
            data_var.sort()
            norm  = random.normal(0,2,len(data_var))
            norm.sort()
            fig = plt.figure(figsize=(12,8),facecolor='1.0')
            plt.plot(norm,data_var,"o")
            z = np.polyfit(norm,data_var, 1)
            p = np.poly1d(z)
            plt.plot(norm,p(norm),"k--", linewidth=2)
            plt.title("Normal Q-Q plot "+var, size=28)
            plt.xlabel("Theoretical quantiles", size=24)
            plt.ylabel("Expreimental quantiles", size=24)
            plt.tick_params(labelsize=16)
            fig.savefig('visualize/'+var+'.png')

    #heatmap function
    def heatmap_corr(self, dataset):
        #UNCOMMENT FOR PLOTTING CORRELATION MATRIX
        corrmat = dataset.corr()
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=.8, square=True);
        plt.xticks(rotation='vertical')
        plt.yticks(rotation='horizontal')
        plt.tight_layout()
        fig.savefig('visualize/heatmap.png')


#Read in preprocessed data
data_all = pd.read_pickle('preprocessed3.pkl')

# cols_names = data_all.columns
# nandat = data_all.isnull().sum()
#
# cols_names = [x for _,x in sorted(zip(nandat,cols_names))]
# nandat = nandat.sort_values()
#
# total = data_all.shape[0]
# nandat = [(e/total)*100 for e in nandat]
# sorted = range(len(cols_names))
#
# plt.bar(sorted, nandat, color = "#3F5D7D", align = 'edge', width=0.6)
# plt.ylabel('Percentage NaN values')
# plt.xticks(sorted, cols_names, rotation='vertical')
# plt.ylim(0,100)
# plt.tight_layout()
# plt.savefig('visualize/nanvalues.png')
# plt.close()

cols_names = np.asarray(data_all.columns)
nzero = np.asarray(data_all.fillna(0).astype(bool).sum(axis=0))
print(nzero)
print(cols_names)
cols_names = [x for _,x in sorted(zip(nzero,cols_names))]
nzero = np.sort(nzero)
total = data_all.shape[0]
nzero = [(e/total)*100 for e in nzero]
sorted = range(len(cols_names))
plt.bar(sorted, nzero, color = "#3F5D7D", align = 'edge', width=0.6)
plt.xticks(sorted, cols_names, rotation='vertical')
plt.ylabel('Percentage non-zero values')
plt.tight_layout()
plt.savefig('visualize/nzero.png')
plt.close()

func = visualize_functions()
func.heatmap_corr(data_all)

for name in ['cheaper_comps','expensive_comps','cheapest_comp']:
    print(data_all[name].describe())
    print(data_all[name].loc[data_all[name] == 0].shape)
    print(data_all.shape[0] - data_all[name].loc[data_all[name] == 0].shape[0])
