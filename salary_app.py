# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:03:46 2022

@author: 91798
"""
import streamlit as st
string="Salary Prediction Web App"
st.set_page_config(page_title=string,page_icon='💰')
st.title("Salary Prediction Web App")
st.write("""
# Salary Prediction Model
Salary vs Experience
""")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data=pd.read_csv(r"C:\Users\91798\Desktop\nw\SALARY\Salary_Data.csv")

X=data['YearsExperience']
print(X)
Y=data.drop('YearsExperience',axis=1)
print(Y)
plt.scatter(X,Y)
plt.plot(X,Y)


exp=st.sidebar.slider("Experience",1,11,1)

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=5)
lin_reg=LinearRegression()
X_train = np.array(X_train).reshape(-1, 1)
X_test = np.array(X_test).reshape(-1, 1)
lin_reg.fit(X_train,y_train)
y_pred=lin_reg.predict(X_test)
score_model=lin_reg.score(X_test,y_test)

plt.scatter(X_train, y_train)
plt.plot(X_test, y_pred, color="blue", linewidth=1)

y_pred_1=lin_reg.predict([[exp]])
st.write(f"Experience",exp)
st.write(f"Salary", float('%.3f'%(y_pred_1)))

st.write("""
# Scatter Plot
Salary vs. Experience
""")

fig = plt.figure()
plt.scatter(X, Y, alpha=0.5)

plt.xlabel('Experience')
plt.ylabel('Salary')
plt.colorbar()

st.pyplot(fig)