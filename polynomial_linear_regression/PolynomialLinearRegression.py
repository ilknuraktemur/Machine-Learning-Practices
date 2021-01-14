# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 17:13:52 2020

@author: q
"""
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("polynomial+regression.csv",sep=";")

x=df.araba_fiyat.values.reshape(-1,1)
y=df.araba_max_hiz.values.reshape(-1,1)


plt.scatter(x,y)
plt.ylabel("araba_max_hiz")
plt.xlabel("araba_fiyat")
plt.show()

#linear regression=y=b0+b1*x
#multiple linear regression=y=b0+b1*x1+b2*x2

#%% linear regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(x,y)

#predict
y_head=lr.predict(x)
plt.plot(x,y_head,color="red",label="linear")
plt.show()

#polynomial linear regression=y= b0+b1*x+b2*x^2+b3*x^3...
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression=PolynomialFeatures(degree=5)

x_polynomial=polynomial_regression.fit_transform(x)
#%% fit
linear_regression2=LinearRegression()
linear_regression2.fit(x_polynomial,y)

#%%
y_head=linear_regression2.predict(x_polynomial)
plt.plot(x,y_head,color="black",label="poly")
plt.legend()
plt.show()





