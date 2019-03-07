import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

dataset = pd.read_csv("DownloadedData\Machine Learning A-Z Template Folder\Part 1 - Data Preprocessing\Data_Preprocessing\Data.csv")

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

Y = labelencoder_X.fit_transform(Y)

print (Y)