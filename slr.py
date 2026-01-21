import numpy as np 	
import matplotlib.pyplot as plt
import pandas as pd	
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# loading the dataset
data = pd.read_csv("/Users/yaswanthkumarvejandla/Downloads/AI ML Engineer/notes/4.nov/10th slr/Salary_Data.csv")

#  splitting the data into independent and dependent variables
x=data.iloc[:,:-1]
y=data.iloc[:,-1]

# splitting the dataset into training and testing sets (80-20%)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)

# import the model and train the model
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# predict the test set
y_pred = regressor.predict(x_test)

# comparison between orginial vs predicted
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

# Visualize the training set
plt.scatter(x_train, y_train, color='red') 
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# Visualize the test set
plt.scatter(x_test, y_test, color='red') 
plt.plot(x_train, regressor.predict(x_train), color='blue')             
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()  



# Predict salary for 6 and 26 years of experience using the trained model
y_6 = regressor.predict([[6]])
y_26 = regressor.predict([[26]])
print(f"Predicted salary for 6 years of experience: ${y_6[0]:,.2f}")
print(f"Predicted salary for 26 years of experience: ${y_26[0]:,.2f}")   


# Check model performance       
bias = regressor.score(x_train, y_train)
variance = regressor.score(x_test, y_test)
train_mse = mean_squared_error(y_train, regressor.predict(x_train))
test_mse = mean_squared_error(y_test, y_pred)       


print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")  

# Save the model using pickle
with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as linear_regression_model.pkl")        

import os 
print(os.getcwd())  


from sklearn.metrics import r2_score,mean_squared_error
R2=r2_score(y_test,y_pred)
MSE=mean_squared_error(y_test,y_pred)
#MSE**(1/2)             
rmse=MSE**(1/2)
print(f"R2 Score: {R2:.2f}")
print(f"MSE: {MSE:.2f}")
print(f"RMSE: {rmse:.2f}")


#regression table code
# introduce to OLS & stats.api
from statsmodels.api import OLS
OLS(y_train,x_train).fit().summary()