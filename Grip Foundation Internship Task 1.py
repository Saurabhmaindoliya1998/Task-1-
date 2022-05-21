#!/usr/bin/env python
# coding: utf-8

# In[8]:


#importing Required libraries
import pandas as pd
import numpy as p
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'Inline')


# In[9]:


#Importing Data
data = pd.read_csv("student_scores - student_scores.csv")


# In[6]:


#First Five records
data.head()


# In[7]:


#total number of row and coloum
data.shape


# In[8]:


data.describe()


# In[10]:


#no null values in Dataset
data.isnull().any()


# In[17]:


data.plot(x="Hours", y="Scores",color='g',style= 'o')

plt.title("Hours Vs Scores")
plt.xlabel("Hours")
plt.ylabel("percentage")
plt.show()


# In[10]:


data.plot.bar(x="Hours", y="Scores", color='c',style= 'o')


# In[11]:


data.sort_values(["Hours"],axis=0, ascending=[True],inplace=True)
data.plot.bar(x="Hours", y ="Scores")


# Observations from above graphs-
# 
# As study hours increases scores also increases

# In[13]:


#dividing the data
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values


# In[3]:


from sklearn.model_selection import train_test_split


# In[14]:


#spliting thee data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[15]:


#algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[16]:


#regression line
line = regressor.coef_*x+regressor.intercept_

#Plotting for the test
plt.scatter(X, y,color="c")
plt.plot(X, line);
plt.show()


# # Ready for Testing

# In[18]:


print("ORIGINAL SCORES")
print(y_test)
print("PREDICTED SCORES")
y_pred = regressor.predict(X_test)
print(y_pred)


# In[19]:


datanew = pd.DataFrame({"Actual":y_test, "Predicted":y_pred})
datanew


# What will be the predicted score if a student studies for 9.25 hrs/day?

# In[20]:


hours = [[9.25]]
prediction = regressor.predict(hours)

print(prediction)


# so the predicted score for 9.25 hours of study per day is 91.9

# In[37]:


from sklearn import metrics
print('Mean absolute error = ',metrics.mean_absolute_error(y_test,  y_pred))


# In[38]:


#checking the model

from sklearn.metrics import r2_score
print("R2 Score = ", r2_score(y_test, y_pred))


# In[ ]:




