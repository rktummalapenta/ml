#!/usr/bin/env python
# coding: utf-8

# California Housing Price Prediction .
# Project 1 
# 
# DESCRIPTION
# 
# Background of Problem Statement :
# 
# The US Census Bureau has published California Census Data which has 10 types of metrics such as the population, median income, median housing price, and so on for each block group in California. The dataset also serves as an input for project scoping and tries to specify the functional and nonfunctional requirements for it.
# 
# Problem Objective :
# 
# The project aims at building a model of housing prices to predict median house values in California using the provided dataset. This model should learn from the data and be able to predict the median housing price in any district, given all the other metrics.
# 
# Districts or block groups are the smallest geographical units for which the US Census Bureau
# publishes sample data (a block group typically has a population of 600 to 3,000 people). There are 20,640 districts in the project dataset.
# 
# Domain: Finance and Housing
# 
# Analysis Tasks to be performed:
# 
# 1. Build a model of housing prices to predict median house values in California using the provided dataset.
# 
# 2. Train the model to learn from the data to predict the median housing price in any district, given all the other metrics.
# 
# 3. Predict housing prices based on median_income and plot the regression chart for it.
# 
# 1. Load the data :
# 
# Read the “housing.csv” file from the folder into the program.
# Print first few rows of this data.
# Extract input (X) and output (Y) data from the dataset.
# 2. Handle missing values :
# 
# Fill the missing values with the mean of the respective column.
# 3. Encode categorical data :
# 
# Convert categorical column in the dataset to numerical data.
# 4. Split the dataset : 
# 
# Split the data into 80% training dataset and 20% test dataset.
# 5. Standardize data :
# 
# Standardize training and test datasets.
# 6. Perform Linear Regression : 
# 
# Perform Linear Regression on training data.
# Predict output for test dataset using the fitted model.
# Print root mean squared error (RMSE) from Linear Regression.
#             [ HINT: Import mean_squared_error from sklearn.metrics ]
# 
# 7. Bonus exercise: Perform Linear Regression with one independent variable :
# 
# Extract just the median_income column from the independent variables (from X_train and X_test).
# Perform Linear Regression to predict housing values based on median_income.
# Predict output for test dataset using the fitted model.
# Plot the fitted model for training data as well as for test data to check if the fitted model satisfies the test data.
# 
# Dataset Description :
# 
# Field	Description
# 1. longitude	(signed numeric - float) : Longitude value for the block in California, USA
# 2. latitude	(numeric - float ) : Latitude value for the block in California, USA
# 3. housing_median_age	(numeric - int ) : Median age of the house in the block
# 4. total_rooms	(numeric - int ) : Count of the total number of rooms (excluding bedrooms) in all houses in the block
# 5. total_bedrooms	(numeric - float ) : Count of the total number of bedrooms in all houses in the block
# 6. population	(numeric - int ) : Count of the total number of population in the block
# 7. households	(numeric - int ) : Count of the total number of households in the block
# 8. median_income	(numeric - float ) : Median of the total household income of all the houses in the block
# 9. ocean_proximity	(numeric - categorical ) : Type of the landscape of the block [ Unique Values : 'NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'  ]
# 10. median_house_value	(numeric - int ) : Median of the household prices of all the houses in the block
#  
# 
# Dataset Size : 20640 rows x 10 columns

# In[43]:


#import the necessary libraries required 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#%matplotlib notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../project/" directory.

import os
print(os.listdir("../project"))


# ### Loading Housing Dataset and Exploratory Data Analysis

# In[89]:


# Import housing excel dataset

df = pd.read_excel(os.path.join(pathlib.Path('.').parent.resolve(),'1553768847_housing.xlsx'),engine='openpyxl')


# In[110]:


print(f"Number of rows and colums and also called shape of the matrix: {df.shape}")
print(f"Columns are \n {df.columns}")


# In[111]:


# Data Set info to understand the columns, dtypes
df.info()


# In[92]:


# Describe to basic statistics of dataset
df.describe().T


# In[93]:


# Showing first five rows in raw dataset
df.head()


# In[94]:


#Display of each column distribution in Scatter Matrix
plt.style.use('ggplot')
fig = plt.figure()
scatter_matrix(df,figsize =(25,25),alpha=1.0,diagonal="kde",marker=".");


# In[95]:


# Column level Histogram
df.hist(figsize=(25,25),bins=50)


# In[97]:


# Finding Correlation using corr function to see which features has impact on the target variable
corr = df.corr()
corr.style.background_gradient()


# ### Handling Missing Values in Dataset

# In[98]:


# Filling total_bedrooms column nan values with mean value of that column
df['total_bedrooms'].fillna(np.mean(df['total_bedrooms']),inplace=True)


# In[99]:


# Data Set info to understand the columns, dtypes after filling NAN values
df.info()


# ### Encoding Categorical Data which ocean_proximity column

# In[100]:


# Converting category data into numerical data
df['ocean_proximity'] = df['ocean_proximity'].astype('category')
df['ocean_proximity'] = df['ocean_proximity'].cat.codes
df['ocean_proximity'].value_counts()


# In[101]:


# Displaying first 5 records after encoding categorical data and filled missing values
df.head()


# ### Split the data into 80% training dataset and 20% test dataset.

# In[102]:


# Extracting X input and Y target to train model

X_data = df.drop('median_house_value', axis=1)
Y_target = df['median_house_value']


# In[103]:


# Display of first 5 rows in X input
X_data.head()


# In[104]:


# Display of first 5 rows in Y target
Y_target.head()


# In[105]:


# Splitting X and Y to train (80%) and test (20%)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_data,Y_target,random_state=1,test_size=0.20)
print("x_train shape {} and size {}".format(x_train.shape,x_train.size))
print("x_test shape {} and size {}".format(x_test.shape,x_test.size))
print("y_train shape {} and size {}".format(y_train.shape,y_train.size))
print("y_test shape {} and size {}".format(y_test.shape,y_test.size))


# ### Standardize training and test datasets.

# In[106]:


# Using StandardScaler to standardize dataset to have zero mean and one Standard Deviation
from sklearn.preprocessing import StandardScaler
# the scaler object (model)
scaler = StandardScaler()# fit and transform the data
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test) 

print("train data")
print(X_train[0:5,:])
print("test data")
print(X_test[0:5,:])


# ### Perform Linear Regression
# 
# ### Median House Values are continuous so its a regression problem

# In[112]:


# Intantiate LinearRegression model from sklearn and training the model with X_train (80%) of input dataset.
from sklearn.linear_model import LinearRegression

linreg = LinearRegression(n_jobs=-1)

#fit the model to the training data
linreg.fit(X_train,y_train)


#print the intercept and coefficients 
print(f"Intercept is {linreg.intercept_}")
print(f"coefficients  is {linreg.coef_}")


# In[113]:


# Predicting the median house values using X_test
y_pred = linreg.predict(X_test)


# In[114]:


print(f"Length of y_pred: {len(y_pred)}")
print(f"Length of y_test: {len(y_test)}")
print(f"First 5 y_pred values: {y_pred[0:5]}")
print(f"First 5 y_test values: {y_test.values[0:5]}")


# In[115]:


# Displaying the Actual and Prediction charts

test = pd.DataFrame({'Predicted':y_pred,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:100])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind='reg',);


# ### Print root mean squared error (RMSE) from Linear Regression.

# In[118]:


from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# ## Bonus exercise: Perform Linear Regression with one independent variable
# 
# 1. Extract just the median_income column from the independent variables (from X_train and X_test). 
# 
# 2. Perform Linear Regression to predict housing values based on median_income. 
# 
# 3. Predict output for test dataset using the fitted model. 
# 4. Plot the fitted model for training data as well as for test data to check if the fitted model satisfies the test data

# In[122]:


# Extracting median_income independent variable which high impact(68.8%) on target variable
X_ind = df['median_income']
# Display of first 5 rows in X input
X_ind.head()


# In[136]:


# Splitting X and Y to train (80%) and test (20%)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_ind,Y_target,random_state=42,test_size=0.20)
print("x_train shape {} and size {}".format(x_train.shape,x_train.size))
print("x_test shape {} and size {}".format(x_test.shape,x_test.size))
print("y_train shape {} and size {}".format(y_train.shape,y_train.size))
print("y_test shape {} and size {}".format(y_test.shape,y_test.size))


# In[148]:


# Intantiate LinearRegression model from sklearn and training the model with X_train (80%) of input dataset.

linreg = LinearRegression()


#fit the model to the training data
# Reshaping due to single column
linreg.fit(np.array(x_train).reshape(-1,1),y_train)


#print the intercept and coefficients 
print(f"Intercept is {linreg.intercept_}")
print(f"coefficients  is {linreg.coef_}")


# In[150]:


# Predicting the median house values using X_test
y_pred = linreg.predict(np.array(x_test).reshape(-1,1))


# In[151]:


print(f"Length of y_pred: {len(y_pred)}")
print(f"Length of y_test: {len(y_test)}")
print(f"First 5 y_pred values: {y_pred[0:5]}")
print(f"First 5 y_test values: {y_test.values[0:5]}")


# In[152]:


# Displaying the Actual and Prediction charts

test = pd.DataFrame({'Predicted':y_pred,'Actual':y_test})
fig= plt.figure(figsize=(16,8))
test = test.reset_index()
test = test.drop(['index'],axis=1)
plt.plot(test[:100])
plt.legend(['Actual','Predicted'])
sns.jointplot(x='Actual',y='Predicted',data=test,kind='reg',);


# In[153]:


from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[155]:


fig = plt.figure(figsize=(25,8))
plt.scatter(y_test,y_pred,marker="o",edgecolors ="r",s=60)
plt.scatter(y_train,linreg.predict(np.array(x_train).reshape(-1,1)),marker="+",s=50,alpha=0.5)
plt.xlabel(" Actual median_house_value")
plt.ylabel(" Predicted median_house_value")


# In[ ]:




