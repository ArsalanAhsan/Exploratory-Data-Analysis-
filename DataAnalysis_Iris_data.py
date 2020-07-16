#!/usr/bin/env python
# coding: utf-8

# In[81]:


# Importing the libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[82]:


#load CSV file
df = pd.read_csv('C:\\Users\\ahsan\\OneDrive\\Desktop\\Iris.csv')
df.head()


# In[83]:


#drop Id column
df = df.drop('Id',axis=1)
df.head()


# ### i) How many data points are there in this data set?
# 

# In[84]:


df.info()


# ### ii) What is the shape of the data?

# In[85]:


df.shape


# ### iii) What are the data types of the columns?
# 

# In[86]:


df.dtypes


# ### iv) What are the column names?(The column names correspond to flower species names, as well as four basic measurements one can make of a flower: the width and length of its petals and the width and length of its sepal (the part of the pant that supports and protects the flower itself)).

# In[87]:


df.columns


# ### v) How many species of flower are included in the data?
# 

# In[88]:


df.Species.unique()


# 
# ### vi) What are the first 10 rows of the data?
# 

# In[89]:


df.head(10)


# ### 3)The dataset that you have downloaded contains errors. Using 1-indexing, these errors are in the 35th and 38th rows. The 35th row should read 4.9,3.1,1.5,0.2,”Irissetosa”, where the fourth feature is incorrect as it appears in the file, and the 38th row should read 4.9,3.6,1.4,0.1,”Iris-setosa”, where the second and third features are incorrect as they appear in the file. Check the entries in the csv file, if not correct, your task is to correct these entries in your DataFrame

# In[90]:


df.loc[34,'PetalWidthCm'] = 0.2
df.loc[37,'SepalWidthCm'] = 3.6
df.loc[37,'PetalLengthCm'] = 1.4


# In[91]:


df[34:38]


# ### 4) The iris dataset is commonly used in machine learning as a proving ground for clustering and classification algorithms. Some researchers have found it useful to use two additional features, called Petal ratio and Sepal ratio, defined as the ratio of the petal length to petal width and the ratio of the sepal length to sepal width,respectively. Add two columns to you DataFrame corresponding to these two new features. Name these columns Petal.Ratio and Sepal.Ratio, respectively

# In[92]:


df['Petal.Ratio']=df['PetalLengthCm']/df['PetalWidthCm']
df['Sepal.Rato']=df['SepalLengthCm']/df['SepalWidthCm']


# In[93]:


df.head()


# ### 5) Save your corrected and extended iris DataFrame to a csv file called iris_corrected.csv. Please include this file in your submission.

# In[94]:


df.to_csv(r'C:\Users\ahsan\OneDrive\Desktop\iris_corrected.csv', index=False) 


# ### 6) Use a Pandas aggregate operation to determine the mean, median, minimum,maximum and standard deviation of the petal and sepal ratio for each of the three species in the data set. Note: you should be able to get all of these numbers in a single table (indeed, in a single line of code) using a well-chosen group-by or aggregate operation.

# In[95]:


# df.groupby('Species').agg({"Sepal.Rato":['mean','median','max','min','std']})
df.groupby('Species').agg({"Sepal.Rato":['mean','median','max','min','std'],"Petal.Ratio":['mean','median','max','min','std']})


# ### 7) Visualize the Iris Dataset by plotting
# #### a. the histogram of the ‘targets’ with respect to each feature of the dataset
# #### b. the scatter-plot between ‘petal-width’ and ‘all other features’ and scatter-plot between ‘petal-length’ and all other features.
# #### c. Use pandas scatter_matrix for plotting all possible combinations along with the histogram.

# In[96]:


df.hist(figsize=(8,8),edgecolor='black', linewidth=1.2)
plt.show()


# In[97]:


# scatter-plot between ‘petal-width’ and ‘all other features
df.plot.scatter(x='PetalWidthCm',y='SepalLengthCm',c='red');
df.plot.scatter(x='PetalWidthCm',y='SepalWidthCm',c='red');
df.plot.scatter(x='PetalWidthCm',y='PetalLengthCm',c='red');
df.plot.scatter(x='PetalWidthCm',y='Species',c='red');
df.plot.scatter(x='PetalWidthCm',y='Petal.Ratio',c='red')
df.plot.scatter(x='PetalWidthCm',y='Sepal.Rato',c='red');
plt.show();


# In[98]:


# scatter-plot between ‘petal-length’ and all other features
df.plot.scatter(x='PetalLengthCm',y='SepalLengthCm',c='lightblue');
df.plot.scatter(x='PetalLengthCm',y='SepalWidthCm',c='lightblue');
df.plot.scatter(x='PetalLengthCm',y='PetalWidthCm',c='lightblue');
df.plot.scatter(x='PetalLengthCm',y='Species',c='lightblue');
df.plot.scatter(x='PetalLengthCm',y='Petal.Ratio',c='lightblue')
df.plot.scatter(x='PetalLengthCm',y='Sepal.Rato',c='lightblue');
plt.show();


# In[99]:


pd.plotting.scatter_matrix(df,figsize=(10,10));


# ### 8) Split the data into training and testing data

# In[100]:


# Importing metrics for evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[101]:


y = df["Species"]
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm","Petal.Ratio","Sepal.Rato"]
X = df[features]


# In[102]:


from sklearn.model_selection import train_test_split

#splitting our dataset for training and validation
#random_state to shuffle our data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# In[103]:


train_y = pd.get_dummies(train_y)
val_y = pd.get_dummies(val_y)


# ### 9) Implement any Machine learning algorithm (classification e.g. KNN) of your choice to classify a given Iris flowers.

# In[104]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(random_state=1)
random_forest.fit(train_X, train_y)


# In[105]:


from sklearn.metrics import mean_absolute_error

preds = random_forest.predict(val_X)
mae = mean_absolute_error(val_y, preds)

print("Mean absolute error is: {:,.5f}".format(mae * 100))
print(random_forest.score(val_X, val_y) * 100)

