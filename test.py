import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.cluster import KMeans
from warnings import filterwarnings
from sklearn import datasets

dataImport = pd.read_csv('winequality-red.csv')
print("Successfully Imported Data!")
dataImport.head()

print(dataImport.shape)
dataImport.describe(include='all')



