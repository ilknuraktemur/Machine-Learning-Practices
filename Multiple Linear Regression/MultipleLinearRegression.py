# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 16:36:54 2020

@author: q
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df=pd.read_csv("multiple_linear_regression_dataset.csv",sep=";")

x=df.iloc[:,[0,2]].values
y=df.maas.values.reshape(-1,1)

multiple_linear_regression=LinearRegression()
multiple_linear_regression.fit(x,y)

b0=multiple_linear_regression.intercept_
print("b0:",b0)

print("b1,b2",multiple_linear_regression.coef_)

#predict
print(multiple_linear_regression.predict(np.array([[10,35],[5,35]])))

