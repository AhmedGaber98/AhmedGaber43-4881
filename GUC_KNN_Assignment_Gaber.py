#!/usr/bin/env python
# coding: utf-8

# # GUC K-nearest neighbor Classification

# ### Import packages and data set
# #### Import the "Classified data" file 

# In[10]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing

from sklearn import model_selection

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report



# In[11]:


df=pd.read_csv("Classified Data")
df.head()


# ### Scale the features using sklearn.preprocessing package

# **Instantiate a scaler standardizing estimator**

# In[12]:


scaler = preprocessing.StandardScaler().fit(df)
scaler


# **Fit the features data only to this estimator (leaving the TARGET CLASS column) and transform**

# In[13]:


scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


# In[14]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# ### Train/Test split, model fit and prediction

# In[28]:


from sklearn.model_selection import train_test_split
X = df_feat
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                    test_size=0.30, random_state=101)


# In[29]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)


# In[30]:


pred = knn.predict(X_test)
pred


# ### Evaluation of classification quality using Confusion Matrix

# In[31]:


confusion_matrix(y_test,pred)
matrix=plot_confusion_matrix(knn, X_test,y_test)
matrix.ax_.set_title('confusion Matrix', color='white')


# ***Print Misclassification error rate***

# In[32]:


tp, fn, fp, tn = confusion_matrix(y_test,pred,labels=[1,0]).reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)

acc=((tp+tn)/(tp+tn+fp+fn))
error = ((fp+fn)/(tp+tn+fp+fn))
print("accuracy is ",acc)
print("error is ",error)


# # Choosing 'k' using cross validation error
# #### Plot the error rate vs. K Value

# In[36]:


from sklearn.model_selection import cross_val_score
array_x=[]
array_y=[]
array_z=[]
i=1
while(i<30):
    array_x+=[i]
    knnclassifier = KNeighborsClassifier(n_neighbors=i)
    y=cross_val_score(knnclassifier, X_train, y_train, cv=10, scoring ='accuracy').mean()
    array_z+=[y]
    array_y+=[1-y]
    i+=1
maxx=max(array_z)
index_max = np.argmax(array_z)+1
print("max k value=" ,index_max)
plt.plot(array_x, array_y)
 
plt.xlabel('k-value')
plt.ylabel('error rate')
plt.show()  


# In[ ]:




