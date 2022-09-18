# Denta Bramasta Hidayat
# 5025201116 / KK IUP 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from warnings import filterwarnings
from sklearn import datasets

dataImport = pd.read_csv('winequality-red.csv')
print("Successfully Imported Data!")
dataImport.head()
print(dataImport.head()) # show all the datas
dataImport.shape

dataImport.dtypes
# print(dataImport.dtypes) # show data type inside of data
dataImport.describe(include='all')
print(dataImport.describe(include='all')) #describe all of datas
dataImport.isnull().any().any() # check missing data
dataImport.applymap(np.isreal)
print(dataImport.applymap(np.isreal))

visual = sns.pairplot(dataImport)
plt.show()
# print(visual)
