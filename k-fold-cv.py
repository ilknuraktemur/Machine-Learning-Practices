from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

#%%
iris=load_iris()

x=iris.data
y=iris.target

#%%
x=(x-np.min(x))/(np.max(x)-np.min(x))

#%% train-test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.3,random_state=42)


#%%
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3) # k=n_neighbors

#%%k-fold CV k=10
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator = knn, X = x_train, y = y_train, cv = 10)
print("avarage accuracy:",np.mean(accuracies))
print("avarage std:",np.std(accuracies))

#%%
knn.fit(x_train,y_train)
print("test accuracy:",knn.score(x_test,y_test))

#%%grid search cross validation

from sklearn.model_selection import GridSearchCV

grid={"n_neighbors":np.arange(1,50)}
knn=KNeighborsClassifier()

knn_cv=GridSearchCV(knn, grid, cv = 10) #GridSearchCV
knn_cv.fit(x,y)

#%%print hyperparameter KNN algroritmasındaki K değeri
print("tuned hyperparameter K:",knn_cv.best_params_)
print("tuned parametreye göre en iyi accuracy(best score):",knn_cv.best_score_)

#%%Grid Search CV with Logistic Regression

x = x[:100,:]
y=y[:100]
from sklearn.linear_model import LogisticRegression

grid={"C":np.logspace(-3,3,7),"penalty":["l1","l2"]} #l1=losso ve l2=ridge

logreg=LogisticRegression(solver="liblinear")
logreg_cv=GridSearchCV(logreg,grid,cv = 10)
logreg_cv.fit(x,y)

print("tuned hyperparameters: (best parameters):",logreg_cv.best_params_)
print("accuracy:",logreg_cv.best_score_)

#%%ÖDEV
x = x[:100,:]
y=y[:100]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.3,random_state=42)

from sklearn.linear_model import LogisticRegression

grid={"C":np.logspace(-3,3,7),"penalty":["l1","l2"]} #l1=losso ve l2=ridge

logreg=LogisticRegression(solver="liblinear")
logreg_cv=GridSearchCV(logreg,grid,cv = 10)
logreg_cv.fit(x_train,y_train)

print("tuned hyperparameters: (best parameters):",logreg_cv.best_params_)
print("accuracy:",logreg_cv.best_score_)

#%%

from sklearn.linear_model import LogisticRegression
logreg2=LogisticRegression(C=1,penalty="l1")
logreg2.fit(x_train,y_train)

print("score lojistik:",logreg2.score(x_test,y_test))





