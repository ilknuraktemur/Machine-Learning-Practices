# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 22:23:22 2020

@author: q
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("decision+tree+regression+dataset.csv",sep=";",header=None)

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

# %%decision tree regression
from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor() #random sate=0
tree_reg.fit(x,y) #ağaç modelini oluşturduk

tree_reg.predict([[6]])

x_=np.arange(min(x),max(x),0.0000001).reshape(-1,1)
y_head=tree_reg.predict(x_)

#%% visualize
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("Tribun level")
plt.ylabel("Ücret")
plt.show()

