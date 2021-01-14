# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:24:17 2020

@author: q
"""

import pandas as pd
#%% import twitter data
data=pd.read_csv("gender_classifier.csv",encoding="latin1")
data=pd.concat([data.gender,data.description],axis=1) #sadece 2 columnu kullanacağım için ikisini birleştirdim.
data.dropna(axis=0,inplace=True) #non olan satırlar silindi.
data.gender=[1 if each=="female" else 0 for each in data.gender]

#%%cleaning data
#regular expression(RE)
import re
first_description=data.description[4]
description=re.sub("[^a-zA-Z]"," ",first_description) #a-z ve A-Z arasında olmayan karakterleri bul ve yerlerine space koy.
description=description.lower() #büyük harften küçük harfe çevirme.

#%%stopwords(irrelevant words)->Gereksiz Kelimeler
import nltk #natural language tool kit
nltk.download("stopwords") #carpus klasörüne stopwords ler indiriliyor.
from nltk.corpus import stopwords #sonra ben carpus klosöründen import ediyorum.

#description=description.split()
#split yerine tokenizer kullanabiliriz.
description=nltk.word_tokenize(description)# split kullanırsak "shouldn't" gibi kelimeler "should" ve "not" olarak ikiye ayrılmaz ama word tokenizer kullanırsak ayrılır.

#gereksiz kelimeleri çıkar
description=[word for word in description if not word in set(stopwords.words("english"))]

#lemmazation
import nltk as nlp
lemma=nlp.WordNetLemmatizer()
description=[lemma.lemmatize(word) for word in description]
description=" ".join(description)

#%%BÜTÜN TEXTLERE DATA CLEAN UP İŞLEMİ
description_list=[]
for description in data.description:
    description=re.sub("[^a-zA-Z]"," ",description)
    description=description.lower()
    description=nltk.word_tokenize(description)
    #description=[word for word in description if not word in set(stopwords.words("english"))]
    lemma=nlp.WordNetLemmatizer()
    description=[lemma.lemmatize(word) for word in description]
    description=" ".join(description)
    description_list.append(description)
    
#BAG OF WORDS
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer=CountVectorizer(stop_words="english")
sparse_matrix=count_vectorizer.fit_transform(description_list).toarray()



#%%model oluşturma
y=data.iloc[:,0]
x=sparse_matrix
#train-test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)
#%%naive bayes
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)

#prediction
y_pred=nb.predict(x_test)
print("accuracy:",nb.score(y_pred.reshape(-1,1),y_test))    
    
    
    
                               




