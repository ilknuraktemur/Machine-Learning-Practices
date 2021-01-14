import pandas as pd
import numpy as np

#%% veriyi al
data=pd.read_csv("data.csv")
#veriden 'id' ve 'Unnamed: 32' columnlarını çıkar
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
#%%
## object tipindeki veriyi int'e çevir.
data.diagnosis=[1 if each=='M' else 0 for each in data.diagnosis]
#x ve y'yi belirle
x_data=data.drop(["diagnosis"],axis=1)
y=data.diagnosis.values
#%%normalization
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%% train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=42)
#%%
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)

print("score:",dt.score(x_test,y_test))



