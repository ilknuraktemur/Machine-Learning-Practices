# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 20:30:47 2020

@author: q
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("random+forest+regression+dataset.csv",sep=";",header=None)

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

#%% random forest regression
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x,y)
print("7.8 levelindeki bilet fiyatı:",rf.predict([[7.8]]))

x_=np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head=rf.predict(x_)

#%% visualize
##plt.scatter(x,y,color="red")
##plt.plot(x_,y_head,color="green")
##plt.xlabel("Level")
##plt.ylabel("Ücret")
##plt.show()
#%% evaluation metric
y_head_for_metric=rf.predict(x)
from sklearn.metrics import r2_score
print("r_score:",r2_score(y,y_head_for_metric))



