#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 19:35:52 2018

@author: Pannu
"""
#Artificial Neural Networks

#Installing Theano
#Installing Tensorflow
#Installing Keras

#Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Part 3 - Evaluating, Improving and Tuning the Model -- EITHER USE THIS OR PART 2! NOT BOTH!
#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
#Use these only if you are going straight into K-fold Cross Validation
from keras.models import Sequential
from keras.layers import Dense
#Function to build the classifier
def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(activation="relu", units=6, input_dim=11, kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation='sigmoid', units=1, kernel_initializer='uniform'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier,batch_size =10, nb_epoch =100)
accuracies= cross_val_score(estimator= classifier, X=X_train, y= y_train, cv = 10, n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()
#Improving the ANN
#Dropout regularization to reduce overfitting if needed

#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
#Use these only if you are going straight into K-fold Cross Validation
from keras.models import Sequential
from keras.layers import Dense
#Function to build the classifier
def build_classifier(optimizer):
    classifier=Sequential()
    classifier.add(Dense(activation="relu", units=6, input_dim=11, kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation='sigmoid', units=1, kernel_initializer='uniform'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[25,32],
              'nb_epoch':[100,500],
              'optimizer':['adam','rmsprop']}
grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring = 'accuracy',
                         cv = 10)
grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_