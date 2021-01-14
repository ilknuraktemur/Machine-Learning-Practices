# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 15:36:03 2020

@author: q
"""

#%% libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% read cvs
data=pd.read_csv("data.csv")
data.drop(["Unnamed: 32","id"],axis=1,inplace=True)
data.diagnosis=[1 if each=="M" else 0 for each in data.diagnosis]
data.info()

y=data.diagnosis.values
x_data=data.drop(["diagnosis"],axis=1)

#%%Normalization
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

#%% train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train2=x_train.T
x_test2=x_test.T
y_train2=y_train.T
y_test2=y_test.T
