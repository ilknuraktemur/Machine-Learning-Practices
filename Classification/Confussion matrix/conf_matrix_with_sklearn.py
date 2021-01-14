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
#%% decision tree
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)

print(" decision tree score:",dt.score(x_test,y_test))

#½½ random forest
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(x_train,y_train)
print("rf algoritma score result",rf.score(x_test,y_test))

#%%
y_pred=rf.predict(x_test)
y_true=y_test
#%% conf matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_true,y_pred)

#%% visualization
import seaborn as sns
import matplotlib.pyplot as plt

f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.2,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()



