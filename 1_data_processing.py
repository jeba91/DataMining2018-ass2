import pandas as pd
import csv
from pandas import HDFStore,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

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
        fig.savefig('visualize/heatmap.png')


#Load the dataset from CSV
try:
    data_all = pd.read_csv("training_set_VU_DM_2014.csv", index_col=0, header=0,  delimiter=",")
except IOError as e:
    print('File not found!')
    raise e


cols_names = data_all.columns

func = visualize_functions()

func.heatmap_corr(data_all)

for col in cols_names:
    print(col)
    print(data_all[col].describe())


# import csv
#
# with open('persons.csv', 'wb') as csvfile:
#     filewriter = csv.writer(csvfile, delimiter=',',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     filewriter.writerow(['Name', 'Profession'])
#     filewriter.writerow(['Derek', 'Software Developer'])
#     filewriter.writerow(['Steve', 'Software Developer'])
#     filewriter.writerow(['Paul', 'Manager'])
