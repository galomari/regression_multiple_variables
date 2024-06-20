#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib as plt
from sklearn import linear_model


# In[2]:


df=pd.read_csv(r"C:\Users\hp\Downloads\homeprices (1).csv")
df.head()


# In[4]:


import math
median_bedrooms=math.floor(df.bedrooms.median())


# In[5]:


median_bedrooms


# In[6]:


df.bedrooms=df.bedrooms.fillna(median_bedrooms)
df


# Again fit makes your model learn, while predict only applies what the model has learned. Thus it makes little sense to use predict before fit.
# 
# When talking about a LinearRegressor, the fit method is the one who will determine the values of the coefficients a and b in the equation y = ax + b of the regressor, according to the training data. Once you have these coefficients, you can use the equation to predict the y value for any x. And if you want to have an idea on it's performance, you'll try x values for which you know the y values (i.e. data in the training set).

# In[11]:


reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)


# In[19]:


reg.coef_


# In[14]:


reg.intercept_


# In[18]:


reg.predict([[3000,3,40]])


# In[20]:


112.06244194*3000+23388.88007794*3+-3231.71790863*40+221323.0018654043


# In[22]:


reg.predict([[2500,4,5]])


# In[23]:


get_ipython().system('pip install word2number')


# In[26]:


df=pd.read_csv(r"C:\Users\hp\Downloads\hiring.csv")
df.head()


# In[30]:


from word2number import w2n
df['experience'] = df['experience'].fillna('zero')
df.experience = df.experience.apply(w2n.word_to_num)


# In[31]:


df.head()


# In[33]:


import math
median_test_score=math.floor(df[['test_score(out of 10)']].median())


# In[34]:


df


# In[39]:


df[['test_score(out of 10)']]=df[['test_score(out of 10)']].fillna(median_test_score)


# In[40]:


df


# In[42]:


reg2=linear_model.LinearRegression()


# In[46]:


reg2.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df[['salary($)']])


# In[48]:


reg2.coef_


# In[50]:


reg.intercept_


# In[51]:


reg.predict([[2,9,6]])


# In[52]:


reg.predict([[12,10,10]])


# In[ ]:




