import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv(r'/Users/yaswanthkumarvejandla/Downloads/AI ML Engineer/notes/4.nov/25th_plr/emp_sal.csv')

x = data.iloc[:,1:2].values
y = data.iloc[:,2:].values

from sklearn.linear_model import LinearRegression
l_reg= LinearRegression()

l_reg.fit(x,y)

plt.scatter(x, y,c='blue')
plt.plot(x,l_reg.predict(x),c ='green')
plt.xlabel('emp postion ')
plt.ylabel(' salary')
plt.show()

from sklearn.preprocessing  import PolynomialFeatures
p_reg=PolynomialFeatures(degree=4)
x_poly = p_reg.fit_transform(x)
p_reg.fit(x_poly,y)

l_reg2=LinearRegression()
l_reg2.fit(x_poly,y)


plt.scatter(x, y, color = 'red')
plt.plot(x, l_reg2.predict(x_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
 

poly_predict= l_reg2.predict(p_reg.fit_transform([[6.5]]))
print(poly_predict)