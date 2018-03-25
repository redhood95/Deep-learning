# self organing map

# importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset

dataset = pd.read_csv('Credit_Card_Applications.csv')

X = dataset.iloc[:,:-1].values
Y= dataset.iloc[:,-1].values

# featue scaling 
from sklearn.preprocessing import MinMaxScaler

sc= MinMaxScaler(feature_range=(0,1))

X=sc.fit_transform(X)

#train som

from minisom import MiniSom

som = MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X, num_iteration=100)

# visualizing the results 

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
marker=['o','s']
color = ['r','g']
for i,x in enumerate(X):
    w = som.winner(X)
    plot(w[0]+0.5,
         w[1]+0.5,
         marker[Y[i]],
         markeredgecolor=color[Y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
    
show()