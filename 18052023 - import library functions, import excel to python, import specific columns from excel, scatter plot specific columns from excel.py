#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm 
sns.set()


# In[3]:


import seaborn as sns


# In[4]:


sns.set()


# In[5]:


data = pd.read_csv('Book2.csv')


# In[6]:


data


# In[7]:


data2 = pd.read_csv('ashby Plot_trial.csv')


# In[ ]:





# In[8]:


data2


# In[9]:


data2


# In[10]:


data2 = pd.read_csv('ashby_Plot trial.csv')


# In[11]:


data2 = pd.read_csv('ashby Plot_trial.csv')


# In[12]:


data2


# In[13]:


df = pd.read_csv('ashby Plot_trial.csv', usecols=['Thickness for Cone'])


# In[14]:


df


# In[15]:


df_clean = df.dropna('df', how='all')


# In[16]:


df_clean = df.dropna('ashby Plot_trial', how='all')


# In[17]:


df


# In[18]:


df2 = pd.read_csv('ashby Plot_trial.csv', usecols=['Figra (pHRR/1s)'])


# In[19]:


df2


# In[20]:


df3 = pd.Dataframe(data2, columns=['Figra (pHRR/1s)','Thickness for Cone'])


# In[21]:


df3.plot (x='Thickness for Cone', y=['Figra (pHRR/1s)'])


# In[22]:


data2.plot (x='Thickness for Cone', y=['Figra (pHRR/1s)'])


# In[23]:


data2.scatter (x='Thickness for Cone', y=['Figra (pHRR/1s)'])


# In[24]:


data2.scatter(x='Thickness for Cone', y=['Figra (pHRR/1s)'])


# In[25]:


plt.scatter (x='Thickness for Cone', y=['Figra (pHRR/1s)'])


# In[26]:


plt.scatter(data2['Thickness for Cone'], data2['Figra (pHRR/1s)'])


# In[27]:


plt.scatter (x='Thickness for Cone', y=['Figra (pHRR/1s)'])
plt.xlabel('Thickness')
plt.ylabel('FIGRA')


# In[28]:


plt.scatter (x='Thickness for Cone', y=['Figra (pHRR/1s)'])

plt.xlabel('Thickness')
plt.ylabel('FIGRA')


# In[29]:


plt.scatter(x='Thickness for Cone', y=['Figra (pHRR/1s)'])

plt.xlabel('Thickness')
plt.ylabel('FIGRA')


# In[30]:


plt.scatter (x='Thickness for Cone', y=['Figra (pHRR/1s)'])


# In[31]:


plt.scatter(data2['Thickness for Cone'], data2['Figra (pHRR/1s)'])


# In[32]:


plt.scatter(data2['Thickness for Cone'], data2['Figra (pHRR/1s)'])

plt.xlabel('Thickness')
plt.ylabel('FIGRA')


# In[7]:


#import library functions
import pandas as pd #data science or data analysis#
import matplotlib.pyplot as plt #plotting tool#
import numpy as np #mathemathical models or functions#
import stasmodel.api as sm #statistic model based on numpy#
import seaborn as sns #python data visualisation library based on matplotlib#
sns.set()#load seaborn default theme and color palette#

#Before importting excel, ensure it is converted to csv
data = pd.read_csv('ashby Plot_trial.csv') # assign csv file with the name ashby Plot_trial to data
data #plot all rows and columns in side ashby Plot_trial
plt.scatter(data['Thickness for Cone'], data2['Figra (pHRR/1s)']) # plotting scatter with thickness as x and FIGRA as y

plt.xlabel('Thickness')
plt.ylabel('FIGRA')


# In[8]:


#Before importting excel, ensure it is converted to csv
data = pd.read_csv('ashby Plot_trial.csv') # assign csv file with the name ashby Plot_trial to data
data #plot all rows and columns in side ashby Plot_trial
plt.scatter(data['Thickness for Cone'], data2['Figra (pHRR/1s)']) # plotting scatter with thickness as x and FIGRA as y

plt.xlabel('Thickness')
plt.ylabel('FIGRA')


# In[9]:


#Before importting excel, ensure it is converted to csv
data = pd.read_csv('ashby Plot_trial.csv') # assign csv file with the name ashby Plot_trial to data
data #plot all rows and columns in side ashby Plot_trial
plt.scatter(data['Thickness for Cone'], data['Figra (pHRR/1s)']) # plotting scatter with thickness as x and FIGRA as y

plt.xlabel('Thickness')
plt.ylabel('FIGRA')


# In[10]:


#Before importting excel, ensure it is converted to csv
data = pd.read_csv('ashby Plot_trial.csv') # assign csv file with the name ashby Plot_trial to data
data 


# In[13]:


#Before importting excel, ensure it is converted to csv
data = pd.read_csv('ashby Plot_trial.csv') # assign csv file with the name ashby Plot_trial to data
plt.scatter(x=data['Thickness for Cone'], x=data['pHRR'],y=data['Heat Flux']  ) # plotting scatter with thickness as x and FIGRA as y

plt.xlabel('Thickness')
plt.ylabel('FIGRA')


# In[15]:


#Before importting excel, ensure it is converted to csv
data = pd.read_csv('ashby Plot_trial.csv') # assign csv file with the name ashby Plot_trial to data
plt.scatter(data['Thickness for Cone'], data['pHRR'],data['Heat Flux']  ) # plotting scatter with thickness as x and FIGRA as y

plt.xlabel('Thickness')
plt.ylabel('FIGRA')


# In[16]:


#Before importting excel, ensure it is converted to csv
data = pd.read_csv('ashby Plot_trial.csv') # assign csv file with the name ashby Plot_trial to data
x1 = data['Thickness for Cone']
x2 = data['pHRR']
y = data['Heat Flux']
plt.scatter(x1, x2, y) # plotting scatter with thickness as x and FIGRA as y

plt.xlabel('Thickness')
plt.ylabel('FIGRA')


# In[17]:


#Before importting excel, ensure it is converted to csv
data = pd.read_csv('ashby Plot_trial.csv') # assign csv file with the name ashby Plot_trial to data
x1 = data['Thickness for Cone']
x2 = data['pHRR']
x3 = data['Heat Flux']
plt.scatter(x1, y) # plotting scatter with thickness as x and FIGRA as y

plt.xlabel('Thickness')
plt.ylabel('FIGRA')


# In[18]:


plt.scatter(x1, x3) # plotting scatter with thickness as x and FIGRA as y


# In[19]:


#Before importting excel, ensure it is converted to csv
data = pd.read_csv('ashby Plot_trial.csv') # assign csv file with the name ashby Plot_trial to data
x1 = data['Thickness for Cone']
x2 = data['pHRR']
x3 = data['Heat Flux']


# In[20]:


plt.scatter(x1, x3) # plotting scatter with thickness as x and FIGRA as y

plt.xlabel('Thickness')
plt.ylabel('FIGRA')


# In[21]:


#Before importting excel, ensure it is converted to csv
data = pd.read_csv('ashby Plot_trial.csv') # assign csv file with the name ashby Plot_trial to data
x1 = data['Thickness for Cone']
x2 = data['pHRR']
x3 = data['Heat Flux']
plt.scatter(x1, x3) # plotting scatter with thickness as x and FIGRA as y
plt.scatter(x2, x3)
plt.xlabel('Thickness')
plt.ylabel('FIGRA')


# In[22]:


#Before importting excel, ensure it is converted to csv
data = pd.read_csv('ashby Plot_trial.csv') # assign csv file with the name ashby Plot_trial to data
x1 = data['Thickness for Cone']
x2 = data['pHRR']
x3 = data['Heat Flux']
plt.scatter(x2, x3) # plotting scatter with thickness as x and FIGRA as y

plt.xlabel('Thickness')
plt.ylabel('FIGRA')


# In[23]:


#Before importting excel, ensure it is converted to csv
data = pd.read_csv('ashby Plot_trial.csv') # assign csv file with the name ashby Plot_trial to data
x2 = data['pHRR']
x3 = data['Heat Flux']
plt.scatter(x3, x1) # plotting scatter with thickness as x and FIGRA as y
plt.scatter(x3, x2)
plt.xlabel('Thickness')
plt.ylabel('FIGRA')


# In[1]:


#Before importting excel, ensure it is converted to csv
data = pd.read_csv('ashby Plot_trial.csv') # assign csv file with the name ashby Plot_trial to data
x1 = data['Thickness']
x2 = data['pHRR']
x3 = data['Heat Flux']
plt.scatter(x3, x1) # plotting scatter with thickness as x and FIGRA as y
plt.scatter(x3, x1)
plt.xlabel('Thickness')
plt.ylabel('FIGRA')


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import pandas as pd

#Before importting excel, ensure it is converted to csv
data = pd.read_csv('ashby Plot_trial.csv') # assign csv file with the name ashby Plot_trial to data
x1 = data['Thickness for Cone']
x2 = data['pHRR']
x3 = data['Heat Flux']
plt.scatter(x3, x2) # plotting scatter with thickness as x and FIGRA as y
plt.scatter(x1, x2)
plt.xlabel('Thickness')
plt.ylabel('FIGRA')


# In[ ]:




