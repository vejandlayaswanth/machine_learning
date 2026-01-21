import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open(r'/Users/yaswanthkumarvejandla/Downloads/AI ML Engineer/projects/slr/linear_regression_model.pkl', 'rb'))

st.title("Simple Linear Regression Salary Prediction")
st.write("Predict Salary based on Years of Experience")    
 
experience = st.number_input(
    "Enter Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.5
)

# Predict button
if st.button("Predict Salary"):
    salary = model.predict([[experience]])
    st.success(f"Predicted Salary: $ {salary[0]:,.2f}")