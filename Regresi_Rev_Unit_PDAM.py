#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("dataset.csv")


# In[3]:


df.head()


# Preprocessing (Normalisasi MinMax Scaler lebih baik dari Standarisasi)

# In[4]:


x = df.drop(['Unit', 'Revenue'], axis=1)
y = df.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
x_ = sc_X.fit_transform(x)

# Reshape data karena hanya satu fitur
y_ = y.reshape(-1,1)
y_ = sc_y.fit_transform(y_)


# In[5]:


x = df.drop(['Unit', 'Revenue'], axis=1)
y = df.iloc[:, -1].values 

from sklearn.preprocessing import MinMaxScaler

# define min max scaler
sc_X = MinMaxScaler()
sc_y = MinMaxScaler()
# transform data
x = sc_X.fit_transform(x)

# Reshape data karena hanya satu fitur
y = y.reshape(-1,1)
y = sc_y.fit_transform(y)
print(x)
print(y)


# Split Data

# In[6]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# PEMODELAN

# Decision Tree Regression
# 

# In[7]:


# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressorDT = DecisionTreeRegressor()
regressorDT.fit(X_train, y_train)


# In[8]:


# Predicting a test data

y_pred_ = regressorDT.predict(X_test)

print(y_pred_)
y_pred_1 = y_pred_.reshape(-1,1)
print(sc_y.inverse_transform(y_pred_1))


# In[9]:


from sklearn.metrics import r2_score

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_pred_ - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_pred_ - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_pred_ , y_test) )


# Random Forest Regression
# 

# In[10]:


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100)
regressor.fit(X_train, y_train)


# In[11]:


# Predicting a test result

y_pred_ = regressor.predict(X_test)


# In[12]:


from sklearn.metrics import r2_score

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_pred_ - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_pred_ - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_pred_ , y_test) )


# Regresi Linear Variabel Jamak

# In[13]:


from sklearn import linear_model
regr = linear_model.LinearRegression()

regr.fit (X_train, y_train)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# In[14]:


# Predicting a test data
y_pred_ = regr.predict(X_test)


# In[15]:


from sklearn.metrics import r2_score
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_pred_ - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_pred_ - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_pred_ , y_test) )


# Super Vector Reggression

# In[16]:


# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor_ = SVR(kernel = 'linear')
#linear = Linear Kernel
regressor_.fit(X_train, y_train)


# In[17]:


# Predicting a test data

y_pred_ = regressor_.predict(X_test)
y_pred_1 = y_pred_.reshape(-1,1)
y_invers = sc_y.inverse_transform(y_pred_1) #untuk invers hasil prediksi ke bentuk asli
print(y_invers)


# In[18]:


from sklearn.metrics import r2_score

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_pred_ - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_pred_ - y_test) ** 2))
print("R2-score: %.2f" % r2_score(y_pred_ , y_test) )


# SAVE MODEL

# In[19]:


import pickle
# Save the Modle to file in the current working directory

Pkl_Filename = "model_regresi_linear2.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(regr, file)


# In[20]:


# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    Pickled_LR_Model = pickle.load(file)

Pickled_LR_Model


# In[21]:


# Use the Reloaded Model to 
# Calculate the accuracy score and predict target values

# Calculate the Score 
score = Pickled_LR_Model.score(X_test, y_test)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

#********
#jika menggunakan data baru (X_test), maka perlu dinormalisasi dahulu X nya
#from sklearn.preprocessing import MinMaxScaler

#sc_X = MinMaxScaler()
#sc_y = MinMaxScaler()
#X_test = sc_X.fit_transform(X_test)
#*********


# Predict the Labels using the reloaded Model
Ypredict = Pickled_LR_Model.predict(X_test)  

y_invers = sc_y.inverse_transform(Ypredict) #untuk invers hasil prediksi ke bentuk asli
print(y_invers)

