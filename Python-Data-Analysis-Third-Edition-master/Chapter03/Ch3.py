#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import pandas library
import pandas as pd

# Create dataframe
sample_data = {'name': ['John', 'Alia', 'Ananya', 'Steve', 'Ben'], 
               'gender': ['M', 'F', 'F', 'M', 'M'], 
               'communcation_skill_score': [40, 45, 23, 39, 39],
               'quantitative_skill_score': [38, 41, 42, 48, 32]}

data = pd.DataFrame(sample_data, columns = ['name', 'gender', 'communcation_skill_score', 'quantitative_skill_score'])
data.head()


# In[2]:


# find mean of communication_skill_score column 
data['communcation_skill_score'].mean(axis=0)


# In[3]:


# find mode of communication_skill_score column
data['communcation_skill_score'].mode()[0]


# In[4]:


# find median of communication_skill_score column
data['communcation_skill_score'].median()


# In[5]:


column_range=data['communcation_skill_score'].max()-data['communcation_skill_score'].min()
print(column_range)


# In[6]:


# First Quartile
q1 = data['communcation_skill_score'].quantile(.25)

# Third Quartile
q3 = data['communcation_skill_score'].quantile(.75)

# Inter Quartile Ratio
iqr=q3-q1
print(iqr)


# In[7]:


# Variance of communication_skill_score
data['communcation_skill_score'].var()


# In[8]:


# Standard deviation of communication_skill_score
data['communcation_skill_score'].std()


# In[9]:


# Describe dataframe
print(data.describe())


# In[127]:


# skewness of communication_skill_score column
data['communcation_skill_score'].skew()


# In[128]:


# kurtosis of communication_skill_score column 
data['communcation_skill_score'].kurtosis()


# In[12]:


# Covariance between columns of dataframe
data.cov()


# In[13]:


# Correlation between columns of dataframe
data.corr(method ='pearson')


# In[ ]:





# ## Parametric Test

# ## One-Sample t-test

# In[15]:


import numpy as np

from scipy.stats import ttest_1samp

data=np.array([63, 75, 84, 58, 52, 96, 63, 55, 76, 83])

mean_value = np.mean(data)

print("Mean:",mean_value)

t_test_value, p_value = ttest_1samp(data, 68)

print("P Value:",p_value)

print("t-test Value:",t_test_value)

# 0.05 or 5% is significance level or alpha.

if p_value < 0.05:    

    print("Hypothesis Rejected")

else:
    
    print("Hypothesis Accepted")


# ## 2-sample t-test

# In[16]:


from scipy.stats import ttest_ind

data1=np.array([63, 75, 84, 58, 52, 96, 63, 55, 76, 83])

data2=np.array([53, 43, 31, 113, 33, 57, 27, 23, 24, 43])

# Compare samples

stat, p = ttest_ind(data1, data2)

print("p-values:",p)

print("t-test:",stat)

# 0.05 or 5% is significance level or alpha.

if p  < 0.05:    

    print("Hypothesis Rejected")

else:

    print("Hypothesis Accepted")    


# ## Paired t-test

# In[22]:


# paired test
from scipy.stats import ttest_rel

# Weights before treatment
data1=np.array([63, 75, 84, 58, 52, 96, 63, 65, 76, 83])

# Weights after treatment
data2=np.array([53, 43, 67, 59, 48, 57, 65, 58, 64, 72])

# Compare weights

stat, p = ttest_rel(data1, data2)

print("p-values:",p)

print("t-test:",stat)

# 0.05 or 5% is the significance level or alpha.

if p  < 0.05:    
    
    print("Hypothesis Rejected")

else:

    print("Hypothesis Accepted")


# ## One-Way ANOVA

# In[25]:


from scipy.stats import f_oneway

# Performance scores of Mumbai location
mumbai=[0.14730927, 0.59168541, 0.85677052, 0.27315387, 0.78591207,0.52426114, 0.05007655, 0.64405363, 0.9825853 , 0.62667439]

# Performance scores of Chicago location
chicago=[0.99140754, 0.76960782, 0.51370154, 0.85041028, 0.19485391,0.25269917, 0.19925735, 0.80048387, 0.98381235, 0.5864963 ]

# Performance scores of London location
london=[0.40382226, 0.51613408, 0.39374473, 0.0689976 , 0.28035865,0.56326686, 0.66735357, 0.06786065, 0.21013306, 0.86503358]

# Compare results using Oneway ANOVA
stat, p = f_oneway(mumbai, chicago, london)

print("p-values:", p)

print("t-test:", stat)


if p  < 0.05:    
    
    print("Hypothesis Rejected")

else:

    print("Hypothesis Accepted")


# In[105]:


# import numpy as np
# np.random.randint(1,120,size=10)


# # Non-Parametric Test

# In[1]:


from scipy.stats import chi2_contingency

# Average performing employees
average=[20, 16, 13, 7]

# Outstanding performing employees
outstanding=[31, 40, 60, 13]

# contingency table
contingency_table= [average, outstanding]

# Apply Test
stat, p, dof, expected = chi2_contingency(contingency_table)

print("p-values:",p)

if p < 0.05:
   print("Hypothesis Rejected")

else:
   print("Hypothesis Accepted")


# ## Mann-Whitney U-test 

# In[27]:


from scipy.stats import mannwhitneyu

# Sample1

data1=[7,8,4,9,8]

# Sample2

data2=[3,4,2,1,1]

# Apply Test

stat, p = mannwhitneyu(data1, data2)

print("p-values:",p)

# 0.01 or 1% is significance level or alpha.

if p  < 0.01:    

    print("Hypothesis Rejected")

else:
    print("Hypothesis Accepted")


# ## Wilcoxon rank test

# In[28]:


from scipy.stats import wilcoxon

# Sample-1
data1 = [1, 3, 5, 7, 9]

# Sample-2 after treatement 
data2 = [2, 4, 6, 8, 10]

# Apply 
stat, p = wilcoxon(data1, data2)

print("p-values:",p)

# 0.01 or 1% is significance level or alpha.

if p  < 0.01:    

    print("Hypothesis Rejected")

else:
    print("Hypothesis Accepted")


# ## Kruskal–Wallis Test

# In[29]:


from scipy.stats import kruskal

# Data sample-1
x = [38, 18, 39, 83, 15, 38, 63,  1, 34, 50]

# Data sample-2
y = [78, 32, 58, 59, 74, 77, 29, 77, 54, 59]

# Data sample-3
z = [117,  92,  42,  79,  58, 117,  46, 114,  86,  26]

# Apply kruskal-wallis test
stat, p = kruskal(x,y,z)

print("p-values:",p)

# 0.01 or 1% is significance level or alpha.

if p  < 0.01:    

    print("Hypothesis Rejected")

else:
    print("Hypothesis Accepted")


# In[ ]:





# In[ ]:




