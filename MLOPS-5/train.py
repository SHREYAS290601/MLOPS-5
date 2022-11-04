import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression

si=SimpleImputer()
oe=OrdinalEncoder()
lr=LinearRegression()


df=pd.read_csv(r'../insurance.csv')
# print(df.head())



df.sex=oe.fit_transform(df.sex.values.reshape(-1,1))
df.smoker=oe.fit_transform(df.smoker.values.reshape(-1,1))
df.region=oe.fit_transform(df.region.values.reshape(-1,1))

x_train,x_test,y_train,y_test=train_test_split(df.drop('charges',axis=1),df.charges,test_size=0.3,random_state=333)

lr.fit(x_train, y_train)

y_pred=lr.predict(x_test)

print(lr.score(x_train, y_train))

if not os.path.isdir('models/'):
    os.mkdir('models') 

file_name='models/model.pkl'
pickle.dump(lr, open(file_name,'wb'))

