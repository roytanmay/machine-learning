# importing libraries  
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
  
#importing datasets  
data_set= pd.read_csv('50_Startups.csv')  
  
#Extracting Independent and dependent Variable  
x= data_set.iloc[:, :-1].values  
y= data_set.iloc[:, 4].values  
  
#Catgorical data  
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = nm.array(ct.fit_transform(x))
  
#Avoiding the dummy variable trap:  
x = x[:, 1:]  
  
  
# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)  
  
#Fitting the MLR model to the training set:  
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_train, y_train)  
  
#Predicting the Test set result;  
y_pred= regressor.predict(x_test)  

import statsmodels.api as sm

x = nm.append(arr = nm.ones((50,1)).astype(int), values=x, axis=1) 

x_opt=x [:, [0,1,2,3,4,5]]  
x_opt = nm.array(x_opt, dtype=float)
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()  
regressor_OLS.summary()