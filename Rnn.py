#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:41:04 2018

@author: yash
"""

# recurrent neural network

# part 1 - data preprocessing

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#importing trainingset

training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:,1:2].values

#feauture scaling 

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()

training_set = sc.fit_transform(training_set)  

#getting inputs and outputs 
X_train = training_set[0:1257]
y_train=training_set[1:1258]

#reshaping

X_train = np.reshape(X_train,(1257, 1, 1))

# part 2 - building the RNN

#importing the keras libraries 

from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import LSTM

#inintialising rnn

regressor = Sequential()

#addong the input and lstm layer

regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))

#adding oputput

regressor.add(Dense(units=1))

#compiling Rnn

regressor.compile(optimizer='adam' ,loss='mean_squared_error')

#fitting the rnn to the training set

#getting real stock price of 2017

test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_set = test_set.iloc[:,1:2].values

#getting the predicted stock price of 2017

inputs = real_set
inputs=sc.transform(inputs)
inputs=np.reshape(inputs,(20, 1, 1))

predicted = regressor.predict(inputs)
predicted = sc.inverse_transform(predicted)

#visualiuzing rnn
plt.plot(real_set,color='red',label='real')
plt.plot(predicted,color='blue',label='predicted')

