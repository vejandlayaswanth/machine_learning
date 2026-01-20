import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt
 
data = pd.read_csv("/Users/yaswanthkumarvejandla/Downloads/AI ML Engineer/notes/4.nov/10th slr/Salary_Data.csv")

x=data.iloc[:,:-1]

y=data.iloc[:,-1]

from  sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

comparison = pd.DataFrame({'actual_x':x_test.iloc[:,0],'actual_y':y_test,'predicted':y_pred})
print(comparison)


plt.scatter(x_test, y_test, color = 'Red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary of employee based on experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


c = regressor.intercept_
print(f'c value : {c}') 

m = regressor.coef_
print(f'm value: {m}')