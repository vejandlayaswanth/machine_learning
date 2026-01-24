import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sn


data=pd.read_csv(r'/Users/yaswanthkumarvejandla/Downloads/AI ML Engineer/notes/4.nov/7th ml data preprocessing/5. Data preprocessing/Data.csv')

x=data.iloc[:,:-1].values
y=data.iloc[:,3].values



from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
x[:,1:3]=imputer.fit_transform(x[:,1:3])




from sklearn.preprocessing import LabelEncoder

le_x= LabelEncoder()

x[:,0]=le_x.fit_transform(x[:,0])


le_y=LabelEncoder()
y=le_y.fit_transform(y)



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)





from sklearn.preprocessing import Normalizer
sc_x= Normalizer()

x_train = sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

