import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure #scale the view
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from warnings import filterwarnings
import time
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