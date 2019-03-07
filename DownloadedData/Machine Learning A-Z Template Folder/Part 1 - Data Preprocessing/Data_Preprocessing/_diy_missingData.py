import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

dataset = pd.read_csv("DownloadedData\Machine Learning A-Z Template Folder\Part 1 - Data Preprocessing\Data_Preprocessing\Data.csv")

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X[:, 1:3])

X[:, 1:3] = imp.transform(X[:, 1:3])

print(X)