import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pickle
from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()
model=pickle.load(open('./models/model.pkl','rb'))

df=pd.read_csv(r'../insurance.csv')
df.sex=oe.fit_transform(df.sex.values.reshape(-1,1))
df.smoker=oe.fit_transform(df.smoker.values.reshape(-1,1))
df.region=oe.fit_transform(df.region.values.reshape(-1,1))

x_test=df.sample(20)
y_pred=model.predict(x_test.drop('charges',axis=1))


