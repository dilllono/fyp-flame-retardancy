#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

