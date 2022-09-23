import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import scipy.cluster.hierarchy as shc
from matplotlib.pyplot import figure #scale the view
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score
from warnings import filterwarnings
from datetime import date

figure(figsize=(16,8), dpi=100)
dataImport = pd.read_csv('winequality-red.csv')
print("successfully added data!")
dataImport.head()
# print(dataImport.head())

# next step is normalize all the data to be the same scale
scale = normalize(dataImport)
scale = pd.DataFrame(scale, columns=dataImport.columns)
scale.head()
print(scale.head())


# SINGLE INTER-CLUSTER DISTANCE
#create dendograms with SINGLE inter-cluster distance
plt.figure(figsize=(16, 8))
plt.title("Dendograms")
dendogram = shc.dendrogram(shc.linkage(scale, method='single'))
plt.axhline(y=6, color='r', linestyle='--')
plt.show()

#create clusters SINGLE inter-cluster distance
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='single')  
cluster.fit_predict(scale)
print(cluster.fit_predict(scale))

# cluster dots
plt.figure(figsize=(16, 8))  
plt.scatter(scale['quality'], scale['alcohol'], c=cluster.labels_) 
plt.show()

# COMPLETE INTER-CLUSTER DISTANCE
#create dendograms with COMPLETE inter-cluster distance
plt.figure(figsize=(16, 8))
plt.title("Dendograms")
dendogram = shc.dendrogram(shc.linkage(scale, method='complete'))
plt.axhline(y=0.5, color='r', linestyle='--')
plt.show()

#create clusters SINGLE inter-cluster distance
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')  
cluster.fit_predict(scale)
print(cluster.fit_predict(scale))

# cluster dots
plt.figure(figsize=(16, 8))  
plt.scatter(scale['quality'], scale['alcohol'], c=cluster.labels_) 
plt.show()