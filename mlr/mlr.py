import numpy as np 
import pandas as pd 
import seaborn as sn

data=pd.read_csv(r'/Users/yaswanthkumarvejandla/Downloads/AI ML Engineer/notes/4.nov/14th mlr dataset/Investment.csv')

x=data.iloc[:,:-1]
y=data.iloc[:,-1]


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
x["State"] = le.fit_transform(x["State"].astype(str))


# another way to create dummies using pandas == pd.get_dummies
#x=pd.get_dummies(x,dtype=int)


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting values 
y_pred=regressor.predict(x_test)


# comparison between actual y values and predicted values 
comparison =pd.DataFrame({'actual':y_test,'predict':y_pred})
print(comparison)


m=regressor.coef_
print(m)

c=regressor.intercept_
print(c)


import statsmodels.api as sm
x_1=x.values
x_1= sm.add_constant(x_1)


x_opt = x_1[:, [0, 1, 2, 3, 4]]   # adjust if your dummy count changes
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_ols.summary())



x_opt = x_1[:, [0, 1, 2, 3]]   # adjust if your dummy count changes
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_ols.summary())



x_opt = x_1[:, [0, 1, 2]]   # adjust if your dummy count changes
regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_ols.summary())





