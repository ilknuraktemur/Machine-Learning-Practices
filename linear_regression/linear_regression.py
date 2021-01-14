# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 03:32:17 2020

@author: q
"""
#import library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#import data
df=pd.read_csv("linear_regression_dataset.csv",sep=";")

#plot data
#plt.scatter(df.deneyim,df.maas)
#plt.xlabel("deneyim")
#plt.ylabel("maas")
#plt.show()

#linear regression

#sklearn library
from sklearn.linear_model import LinearRegression

#linear regression model
linear_reg=LinearRegression()

x=df.deneyim.values.reshape(-1,1)
y=df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

#prediction
b0=linear_reg.predict([[0]])
print("b0:",b0)
b0_=linear_reg.intercept_
print("b0_",b0_) #üsttekiyle aynı olmalı. #y eksenini kestiği nokta

b1=linear_reg.coef_
print("b1:",b1) #eğim(slope)

#maas=1663.89519747+1138.34819698*11
onbir_yillik_maas=1663.89519747+1138.34819698*11
print("11 yıllık maas:",onbir_yillik_maas)

print("11 yıllık maas:",linear_reg.predict([[11]]))

array=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1) #deneyim
plt.scatter(x,y)
plt.show()

y_head=linear_reg.predict(array)  #maas
plt.plot(array,y_head,color="red")

#r-square with linear regression
#from sklearn.metrics import r2_score
#print("r_score:",r2_score(y,y_head))
