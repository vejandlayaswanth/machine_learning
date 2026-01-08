import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
 
data = pd.read_csv(r'/Users/yaswanthkumarvejandla/Downloads/AI ML Engineer/notes/4.nov/6th/Data.csv')

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

# filling  missing values using simple imputer 

from sklearn.impute import  SimpleImputer

impute = SimpleImputer(strategy='median')

impute= impute.fit(x[:,1:3])

x[:,1:3]= impute.transform(x[:,1:3])

# x[:,1:3]=impute.fit_transform(x[:,1:3]) another simple line to write fit and transform code in single line 
  

 # changing categorical data into nerical data by using label encode 

from sklearn.preprocessing import LabelEncoder
 
le = LabelEncoder()
 
x[:,0] = le.fit_transform(x[:,0])

# label encode for y  
le_y= LabelEncoder()
y=le_y.fit_transform(y)


# splitting data into x_train , x_test , y_train , y_test 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


