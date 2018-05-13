import copy
import os
import numpy as np

import math
import pandas
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.random as random
from pandas import DataFrame
from datetime import datetime
import numpy
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


#Read in preprocessed data
data_all = pd.read_pickle('preprocessed.pkl')

print(data_all.describe())
