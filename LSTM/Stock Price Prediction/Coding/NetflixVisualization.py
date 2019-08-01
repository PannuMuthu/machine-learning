# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# Importing the dataset 
dataset = pd.read_csv('netflix.csv')
rows = dataset.values.tolist()

#Initializing variables
X = []
Y = []
x_train = []
x_test = []
y_train = []
y_test = []

    
for row in rows:
    X.append(int(''.join(row[0].split('-'))))
    Y.append(row[6])

# split training and test data  
x_train, x_test, y_train, y_test = train_test_split(X,Y,train_size=0.9,test_size=0.1)


# Relationship between Sale Price and Above Grade Living Area (Outside land)


sns.distplot(dataset['Volume'])
