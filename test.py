# Denta Bramasta Hidayat
# 5025201116 / KK IUP 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure #scale the view
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from warnings import filterwarnings
import time
from datetime import date



figure(figsize=(16, 8), dpi=100) # set the default views
dataImport = pd.read_csv('winequality-red.csv')
print("Successfully Imported Data!")
dataImport.head()
# print(dataImport.head()) # show all the datas
dataImport.shape

dataImport.dtypes
# print(dataImport.dtypes) # show data type inside of data
dataImport.describe(include='all')
# print(dataImport.describe(include='all')) #describe all of datas
dataImport.isnull().any().any() # check missing data
dataImport.applymap(np.isreal)
# print(dataImport.applymap(np.isreal))

visual = sns.pairplot(dataImport)
# plt.show()
# plt.savefig('/Users/dentabramasta/FigureKKWine/', transparent=True)
# print(visual)

visual2 = sns.displot(dataImport['quality'])
# plt.show()
# plt.savefig('test.png', dpi=100)

# set the x and y and array
X = dataImport.iloc[:,[1,2,3,4]]
y = dataImport['quality']
X_scaled = StandardScaler().fit_transform(X)
print(dataImport.head()) 

# Elbow Method
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
f3, ax = plt.subplots(figsize=(16, 8))
plt.plot(range(1,11),wcss)
plt.title('Elbow Method showing clusters')
plt.xlabel('clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters = 2)
start_time = time.time()
clusters = kmeans.fit_predict(X_scaled)
today = date.today()
print("--- %s seconds ---" % (time.time() - start_time))
labels = kmeans.labels_

#Visualization of clustering
colors = 'rgbkcmy'
for i in np.unique(clusters):
    plt.scatter(X_scaled[clusters==i,0],
               X_scaled[clusters==i,1],
               color=colors[i], label='Cluster' + str(i+1))
plt.legend()
plt.show()

