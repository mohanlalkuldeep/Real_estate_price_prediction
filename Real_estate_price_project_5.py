#!/usr/bin/env python
# coding: utf-8

# In[14]:


import scipy


# In[13]:


import seaborn as sns


# In[12]:


import sklearn


# In[11]:


import statsmodels.api as sm


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


data=pd.read_csv(r"C:\Users\nnhp\Downloads\real_estate_price_size_year.csv")
data


# In[7]:


data.describe()


# In[8]:


y= data["price"]
x1= data["size"]


# In[9]:


plt.scatter(x1,y)
plt.xlabel("size", fontsize=20)
plt.ylabel("price", fontsize=20)
plt.show()


# In[15]:


x= sm.add_constant(x1)
model= sm.OLS(y,x)
results= model.fit()
results.summary()


# In[16]:


plt.scatter(x1,y)
yhat= 223.1787*x1 + 101900
fig= plt.plot(x1,yhat,lw=4, c= 'orange', label='regression line')
plt.xlabel('size', fontsize=20)
plt.ylabel('price', fontsize=20)
plt.show()


# In[37]:


new_data = pd.DataFrame({'constant' : 1, 'size':[200,237,543]})


# In[38]:


new_data


# In[39]:


prediction= results.predict(new_data)
prediction


# In[40]:


predictiondf= pd.DataFrame({'prediction': prediction})
joined = new_data.join(predictiondf)
joined.rename(index={0:'a',1:'b',2:'c'})


# In[ ]:




