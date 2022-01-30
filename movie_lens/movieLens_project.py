#!/usr/bin/env python
# coding: utf-8

# ### Movie Lens Project
# ### ************

# ### Problem Objective :
# 
# Here, we ask you to perform the analysis using the Exploratory Data Analysis technique. You need to find features affecting the ratings of any particular movie and build a model to predict the movie ratings.
# 
# Domain: Entertainment
# 
# ### Analysis Tasks to be performed:
# 
# Import the three datasets
# Create a new dataset [Master_Data] with the following columns MovieID Title UserID Age Gender Occupation Rating. (Hint: (i) Merge two tables at a time. (ii) Merge the tables using two primary keys MovieID & UserId)
# Explore the datasets using visual representations (graphs or tables), also include your comments on the following:
# 1. User Age Distribution
# 2. User rating of the movie “Toy Story”
# 3. Top 25 movies by viewership rating
# 4. Find the ratings for all the movies reviewed by for a particular user of user id = 2696
# 
# 
# ### Feature Engineering:
#             Use column genres:
# 
# 1. Find out all the unique genres (Hint: split the data in column genre making a list and then process the data to find out only the unique categories of genres)
# 2. Create a separate column for each genre category with a one-hot encoding ( 1 and 0) whether or not the movie belongs to that genre. 
# 3. Determine the features affecting the ratings of any particular movie.
# 4. Develop an appropriate model to predict the movie ratings

# In[64]:


# Import required libraries
import os
import pathlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


print(f"Numpy Version: {np.__version__}")
print(f"Pandas Version: {pd.__version__}")


# ####### Import the three datasets 

# In[3]:


# Import movies, users and ratings datasets

path = os.path.join(pathlib.Path('.').parent.resolve(),'Data science with Python 1')

# Format - MovieID::Title::Genres
movies_data = pd.read_csv(os.path.join(path,'movies.dat'),sep='::', names=['MovieID', 'Title', 'Genres'],encoding='latin-1')

# Format - UserID::MovieID::Rating::Timestamp
ratings_data = pd.read_csv(os.path.join(path,'ratings.dat'),sep='::', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

# Format -  UserID::Gender::Age::Occupation::Zip-code
users_data = pd.read_csv(os.path.join(path,'users.dat'),sep='::', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])


# In[4]:


# Describing all datasets

print("Movies Data:")
print(movies_data.describe())
print(movies_data.info)


# In[5]:


# Describing all datasets

print("Ratings Data:")
print(ratings_data.describe())
print(ratings_data.info)


# In[7]:


# Describing all datasets

print("Users Data:")
print(users_data.describe())
print(users_data.info)


# #######
# 
# Create a new dataset [Master_Data] with the following columns MovieID Title UserID Age Gender Occupation Rating. (Hint: (i) Merge two tables at a time. (ii) Merge the tables using two primary keys MovieID & UserId)

# In[6]:


print(movies_data.head(1))
print(users_data.head(1))
print(ratings_data.head(1))


# In[7]:


movies_ratings = ratings_data.merge(movies_data, on='MovieID', how='left')


# In[8]:


movies_ratings_users = movies_ratings.merge(users_data, on='UserID', how='left')


# In[36]:


data = movies_ratings_users[['MovieID', 'Title', 'UserID', 'Genres', 'Age', 'Gender', 'Occupation', 'Zip-code', 'Timestamp', 'Rating']]


# In[37]:


data.columns


# In[38]:


data


# #### Explore the datasets using visual representations (graphs or tables), also include your comments on the following: 
#     
# 1. User Age Distribution 
# 2. User rating of the movie “Toy Story” 
# 3. Top 25 movies by viewership rating 
# 4. Find the ratings for all the movies reviewed by for a particular user of user id = 2696

# ### Cleaning Data Set dropping NA values etc

# In[39]:


data.info()


# In[41]:


data.describe().T


# In[42]:


# Dropping NA rows
data.dropna()


# In[43]:


data.hist(column='Age')


# In[44]:


bins_list = [1, 18, 25, 35, 45, 50, 56]


# In[45]:


data.hist(column='Age', bins=bins_list)


# 
# User rating on the movie Toy Story¶
# 
# Toy story is MovieID = 1, let's get a distribution of ratings on that single ID
# 

# In[46]:


data[data['MovieID']==1].hist(column='Rating')


# In[47]:


data[data['MovieID'] == 1].count()


# In[48]:


data[data['MovieID'] == 1].Rating.mean()


# In[49]:


data.groupby('Title').size().sort_values(ascending=False)[:25]


# In[50]:


user_2696 = data[data['UserID']==2696]


# In[51]:


user_2696.shape


# In[52]:


user_2696.sort_values(by='Rating', ascending=False)


# In[53]:


data['Genre_list'] = data['Genres'].apply(lambda x: x.split('|'))


# In[55]:


data['Gender'].replace(['F','M'],[0,1],inplace=True)


# In[ ]:





# In[62]:


from sklearn.preprocessing import MultiLabelBinarizer

## assign a new series to the genres_list column that contains a list of categories for each movie
list2series = pd.Series(data.Genre_list)

mlb = MultiLabelBinarizer()

## use mlb to create a new dataframe of the genres from the list for each row from the original data

one_hot_genres = pd.DataFrame(mlb.fit_transform(list2series),columns=mlb.classes_,index=list2series.index)


# In[63]:


print(one_hot_genres.head())


# In[66]:


features = data[['MovieID', 'Rating', 'Age', 'Gender', 'Occupation', 'Zip-code']]


# In[68]:


features.corr()['Rating']


# In[69]:


master_features = pd.merge(features, one_hot_genres, left_index=True, right_index=True)


# In[71]:


master_features.head()


# In[73]:


master_features.drop('Zip-code',axis=1,inplace=True)


# In[74]:


master_features.head()


# In[75]:


master_features.corr()


# In[76]:


X_data = master_features.drop(['MovieID','Rating'], axis=1)

Y_target = master_features['Rating']


# In[77]:


X_data.shape


# In[78]:


Y_target.shape


# In[79]:


X_data.head()


# In[80]:


Y_target.head()


# In[82]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_data,Y_target,random_state=1,test_size=0.25)


# In[83]:


from sklearn.linear_model import LogisticRegression


# In[84]:


logreg = LogisticRegression(max_iter=100000)


# In[85]:


logreg.fit(x_train,y_train)


# In[86]:


y_pred = logreg.predict(x_test)


# In[87]:


from sklearn import metrics
metrics.accuracy_score(y_test,y_pred)


# In[88]:


# print the first 30 true and predicted responses
print ('actual:    ', y_test.values[0:30])
print ('predicted: ', y_pred[0:30])


# In[ ]:




