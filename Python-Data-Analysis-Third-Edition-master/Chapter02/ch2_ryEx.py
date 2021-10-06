#!/usr/bin/env python
# coding: utf-8

# ## Understanding NumPy Array

# In[ ]:


# Creating an array
import numpy as np
a= np.array([2,4,6,8,10])
print(a)


# In[ ]:


# Creating an array using arange()
import numpy as np
a= np.arange(1,11)
print(a)


# In[ ]:


import numpy as np

p= np.zeros((3,3))   # Create an array of all zeros
print(p) 

q= np.ones((2,2))    # Create an array of all ones
print(q)

r= np.full((2,2), 4)  # Create a constant array
print(r) 

s= np.eye(4)         # Create a 2x2 identity matrix
print(s) 

t= np.random.random((3,3))  # Create an array filled with random values
print(t)


# In[ ]:


# Creating an array using arange()
import numpy as np
a= np.arange(1,11)
print(type(a))
print(a.dtype)


# In[ ]:


# check shape pf Array
print(a.shape)


# In[ ]:


a = np.array([[5,6],[7,8]])
print(a)


# In[ ]:


print(a[0,0])


# In[ ]:


print(a[0,1])


# In[ ]:


print(a[1,0])


# In[ ]:


print(a[1,1])


# ## NumPy Array Numerical Data Types

# In[ ]:


print(np.float64(21))


# In[ ]:


print(np.int8(21.0)) 


# In[ ]:


print(np.bool(21))


# In[ ]:


print(np.bool(0)) 


# In[ ]:


print(np.bool(21.0)) 


# In[ ]:


print(np.float(True)) 


# In[ ]:


print(np.float(False)) 


# In[ ]:


arr=np.arange(1,11, dtype= np.float32)

print(arr)


# In[ ]:


np.int(42.0 + 1.j)


# In[ ]:


c= complex(42, 1)
print(c)


# In[ ]:


print(c.real,c.imag)


# In[ ]:


# Creating an array
import numpy as np
a = np.array([2,4,6,8,10])

print(a.dtype)


# In[ ]:


print(a.dtype.itemsize)


# In[ ]:


# Create numpy array using arange() function
var1=np.arange(1,11, dtype='int')

print(var1)


# In[ ]:


print(np.arange(0,2, dtype='bool'))


# In[ ]:


print(np.dtype(float))


# In[ ]:


print(np.dtype('f'))


# In[ ]:


print(np.dtype('d')) 


# In[ ]:


print(np.dtype('f8'))


# In[ ]:


var2=np.array([1,2,3],dtype='float64')

print(var2.dtype.char)


# In[ ]:


print(var2.dtype.type)


# ## Manipulating Shape of NumPy Array

# In[ ]:


# Create an array
arr = np.arange(12)


# In[ ]:


# Reshape the array dimension
new_arr=arr.reshape(4,3)

print(new_arr)


# In[ ]:


# Reshape the array dimension
new_arr2=arr.reshape(3,4)

print(new_arr2)


# In[ ]:


# Create an array
arr=np.arange(1,10).reshape(3,3)
print(arr)


# In[ ]:


# flatten the array
print(arr.flatten())


# In[ ]:


# ravel() function 
print(arr.ravel())


# In[ ]:


# Transpose the matrix
print(arr.transpose())


# In[ ]:


# resize the matrix
arr.resize(1,9)
print(arr)


# ## Stacking of Numpy arrays

# In[ ]:


arr1 = np.arange(1,10).reshape(3,3)
print(arr1)


# In[ ]:


arr2 = 2*arr1
print(arr2)


# In[ ]:


arr3=np.hstack((arr1, arr2))

print(arr3)


# In[ ]:


# Horizontal stacking using concatenate() function
arr4=np.concatenate((arr1, arr2), axis=1)
print(arr4)


# In[ ]:


arr5=np.vstack((arr1, arr2))
print(arr5)


# In[ ]:


arr6=np.concatenate((arr1, arr2), axis=0) 
print(arr6)


# In[ ]:


arr7=np.dstack((arr1, arr2))
print(arr7)


# In[ ]:


# Create 1-D array
arr1 = np.arange(4,7) 
print(arr1)


# In[ ]:


# Create 1-D array
arr2 = 2 * arr1
print(arr2)


# In[ ]:


# Create column stack
arr_col_stack = np.column_stack((arr1,arr2))
print(arr_col_stack)


# In[ ]:


# Create row stack
arr_row_stack = np.row_stack((arr1,arr2)) 
print(arr_row_stack)


# ## Partitioning Numpy Array

# In[ ]:


# Create an array
arr=np.arange(1,10).reshape(3,3)
print(arr)


# In[ ]:


# Peroform horizontal splitting
arr_hor_split=np.hsplit(arr, 3)

print(arr_hor_split)


# In[ ]:


# vertical split
arr_ver_split=np.vsplit(arr, 3)

print(arr_ver_split)


# In[ ]:


# split with axis=0
arr_split=np.split(arr,3,axis=0)

print(arr_split)


# In[ ]:


# split with axis=1
np.split(arr,3,axis=1)


# ## Changing Datatype of NumPy Arrays

# In[ ]:


# Create an array
arr=np.arange(1,10).reshape(3,3)
print("Integer Array:",arr)

# Change datatype of array
arr=arr.astype(float)

# print array
print("Float Array:", arr)

# Check new data type of array
print("Changed Datatype:", arr.dtype)


# In[ ]:


# Change datatype of array
arr=arr.astype(float)

# Check new data type of array
print(arr.dtype)


# In[ ]:


# Create an array
arr=np.arange(1,10)

# Convert NumPy array to Python List
list1=arr.tolist()
print(list1)


# ## Creating NumPy views and copies

# In[ ]:


# Create NumPy Array
arr = np.arange(1,5).reshape(2,2)
print(arr)

# Create no copy only assignment
arr_no_copy=arr

# Create Deep Copy
arr_copy=arr.copy()

# Create shallow copy using View
arr_view=arr.view()

print("Original Array: ",id(arr))
print("Assignment: ",id(arr_no_copy))
print("Deep Copy: ",id(arr_copy))
print("Shallow Copy(View): ",id(arr_view))


# In[ ]:


# Update the values of original array
arr[1]=[99,89]

# Check values of array view
print("View Array:\n", arr_view)

# Check values of array copy
print("Copied Array:\n", arr_copy)


# ## Slicing NumPy Array

# In[ ]:


# Create NumPy Array
arr = np.arange(10) 
print(arr)


# In[ ]:


print(arr[3:6])


# In[ ]:


print(arr[3:])


# In[ ]:


print(arr[-3:])


# In[ ]:


print(arr[2:7:2])


# ## Boolean and Fancy Indexing

# In[ ]:


# Create NumPy Array
arr = np.arange(21,41,2)
print("Orignial Array:\n",arr)

# Boolean Indexing
print("After Boolean Condition:",arr[arr>30])


# In[ ]:


# Create NumPy Array
arr = np.arange(1,21).reshape(5,4)
print("Orignial Array:\n",arr)

# Selecting 2nd and 3rd row
indices = [1,2]
print("Selected 1st and 2nd Row:\n", arr[indices])

# Selecting 3nd and 4th row
indices = [2,3]
print("Selected 3rd and 4th Row:\n", arr[indices])


# In[ ]:


# Create row and column indices
row = np.array([1, 2])
col = np.array([2, 3])

print("Selected Sub-Array:", arr[row, col])


# ## Broadcasting arrays

# In[ ]:


# Create NumPy Array
arr1 = np.arange(1,5).reshape(2,2) 
print(arr1)


# In[ ]:


# Create another NumPy Array
arr2 = np.arange(5,9).reshape(2,2) 
print(arr2)


# In[ ]:


# Add two matrices
print(arr1+arr2)


# In[ ]:


# Multiply two matrices
print(arr1*arr2)


# In[ ]:


# Add a scaler value
print(arr1 + 3)


# In[ ]:


# Multiply with a scalar value
print(arr1 * 3)


# ## Create DataFrame

# In[ ]:


# Import pandas library 
import pandas as pd 
# Create empty DataFrame
df = pd.DataFrame() 

# Header of dataframe. 
df.head()


# In[ ]:


df


# In[ ]:


# Create dictionary of list
data = {'Name': ['Vijay', 'Sundar', 'Satyam', 'Indira'], 'Age': [23, 45, 46, 52 ]}   

# Create the pandas DataFrame 
df = pd.DataFrame(data)

# Header of dataframe. 
df.head()


# In[ ]:


# Pandas DataFrame by lists of dicts. 
# Initialise data to lists. 
data =[ {'Name': 'Vijay',  'Age': 23},{'Name': 'Sundar',  'Age': 25},{'Name': 'Shankar',  'Age': 26}]
# Creates DataFrame. 
df = pd.DataFrame(data,columns=['Name','Age']) 
# Print dataframe header 
df.head()  


# In[ ]:


# Creating DataFrame using list of tuples.
data = [('Vijay', 23),( 'Sundar', 45), ('Satyam', 46), ('Indira',52)] 
# Create dataframe
df = pd.DataFrame(data, columns=['Name','Age'])
# Print dataframe header 
df.head()  


# ## Pandas Series

# In[ ]:


# Creating Pandas Series using Dictionary
dict1 = {0 : 'Ajay', 1 : 'Jay', 2 : 'Vijay'}
# Create Pandas Series
series = pd.Series(dict1)
# Show series
series


# In[ ]:


# load Pandas and NumPy
import pandas as pd
import numpy as np
# Create NumPy array
arr = np.array([51,65,48,59, 68])
# Create Pandas Series
series = pd.Series(arr)
series


# In[ ]:


# load Pandas and NumPy
import pandas as pd
import numpy as np
# Create Pandas Series
series = pd.Series(10, index=[0, 1, 2, 3, 4, 5])
series


# In[ ]:





# In[ ]:


# Import pandas 
import pandas as pd

# Load data using read_csv() 
df = pd.read_csv("WHO_first9cols.csv")

# Show initial 5 records
df.head()


# In[ ]:


# Show last 5 records
df.tail()


# In[ ]:


# Show the shape of DataFrame
print("Shape:", df.shape)


# In[ ]:


# Check the column list of DataFrame
print("List of Columns:", df.columns)


# In[ ]:


# Show the datatypes of columns
print("Data types:", df.dtypes)


# In[ ]:


# Select a series
country_series=df['Country']


# In[ ]:


# check datatype of series
type(country_series)


# In[ ]:


print(country_series.index)


# In[ ]:


# Convert Pandas Series into List
print(country_series.values)


# In[ ]:


# Country name
print(country_series.name)


# In[ ]:


# Pandas Series Slicing
country_series[-5:]


# In[ ]:


# Creating Pandas Series using Dictionary
dict1 = {0 : 'Ajay', 1 : 'Jay', 2 : 'Vijay'}
# Create Pandas Series
series = pd.Series(dict1)
# Show series
series


# In[ ]:


# load Pandas and NumPy
import pandas as pd
import numpy as np
# Create NumPy array
arr = np.array([51,65,48,59, 68])
# Create Pandas Series
series = pd.Series(arr)
series


# In[ ]:


# load Pandas and NumPy
import pandas as pd
import numpy as np
# Create Pandas Series
series = pd.Series(10, index=[0, 1, 2, 3, 4, 5])
series


# ## Querying Data

# In[ ]:


get_ipython().system('pip install quandl')


# In[ ]:


import quandl

sunspots = quandl.get("SIDC/SUNSPOTS_A")

sunspots.head()


# In[ ]:


sunspots.head()


# In[ ]:


sunspots.tail()


# In[ ]:


sunspots.columns


# In[ ]:


# Select columns
sunspots_filtered=sunspots[['Yearly Mean Total Sunspot Number','Definitive/Provisional Indicator']]

# Show top 5 records
sunspots_filtered.head()


# In[ ]:


# Select rows using index
sunspots["20020101": "20131231"]


# In[ ]:


# Boolean Filter 
sunspots[sunspots['Yearly Mean Total Sunspot Number'] > sunspots['Yearly Mean Total Sunspot Number'].mean()]


# ## Statistics

# In[ ]:


# Import pandas 
import pandas as pd

# Load data using read_csv() 
df = pd.read_csv("WHO_first9cols.csv")

# Show initial 5 records
df.head()


# In[ ]:


df.shape


# In[ ]:


# Describe the dataset
df.describe()


# In[ ]:


# Count number of observation
df.count()


# In[ ]:


# Compute median of all the columns
df.median()


# In[ ]:


# Compute minimum of all the columns
df.min()


# In[ ]:


# Compute maximum of all the columns
df.max()


# In[ ]:


# Compute standard deviation of all the columns
df.std()


# ## Grouping Pandas DataFrames

# In[ ]:


df.head()


# In[ ]:


# Group By Dataframe on the basis of Continent column
df.groupby('Continent').mean()


# In[ ]:


df.groupby('Continent').mean()['Adult literacy rate (%)']


# ## Joins

# In[ ]:


# Import pandas 
import pandas as pd

# Load data using read_csv() 
dest = pd.read_csv("dest.csv")

# Show DataFrame
dest.head()


# In[ ]:


# Load data using read_csv() 
tips = pd.read_csv("tips.csv")

# Show DataFrame
tips.head()


# In[ ]:


# Join DataFrames using Inner Join
df_inner= pd.merge(dest, tips, on='EmpNr', how='inner')
df_inner.head()


# In[ ]:


# Join DataFrames using Outer Join
df_outer= pd.merge(dest, tips, on='EmpNr', how='outer')
df_outer.head()


# In[ ]:


# Join DataFrames using Right Outer Join
df_right= pd.merge(dest, tips, on='EmpNr', how='right')
df_right


# In[ ]:


# Join DataFrames using Left Outer Join
df_left= pd.merge(dest, tips, on='EmpNr', how='left')
df_left


# ## Missing Values

# In[ ]:


# Import pandas 
import pandas as pd

# Load data using read_csv() 
df = pd.read_csv("WHO_first9cols.csv")

# Show initial 5 records
df.head()


# In[ ]:


# Count missing values in DataFrame
pd.isnull(df).sum()


# In[ ]:


# Count missing values in DataFrame
df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:


# Drop all the missing values
df.dropna(inplace=True)


# In[ ]:


df.info()


# In[ ]:


# Load data using read_csv() 
df = pd.read_csv("WHO_first9cols.csv")

# Show initial 5 records
df.head()


# In[ ]:


df.info()


# In[ ]:


# Fill missing values with 0
df.fillna(0,inplace=True)


# In[ ]:


df.info()


# ## Pivot Table

# In[ ]:


# Import pandas 
import pandas as pd

# Load data using read_csv() 
purchase = pd.read_csv("purchase.csv")

# Show initial 10 records
purchase.head(10)


# In[ ]:


# Summarise dataframe using pivot table
pd.pivot_table(purchase,values='Number', index=['Weather',],
                    columns=['Food'], aggfunc=np.sum)


# ## Dealing with dates

# In[ ]:


# Date range function
pd.date_range('01-01-2000', periods=45, freq='D')


# In[ ]:


# Convert argument to datetime
pd.to_datetime('1/1/1970')


# In[ ]:


# Convert argument to datetime in specified format
pd.to_datetime(['20200101', '20200102'], format='%Y%m%d')


# In[ ]:


# Value Error
pd.to_datetime(['20200101', 'not a date'])


# In[ ]:


# Handle value error
pd.to_datetime(['20200101', 'not a date'], errors='coerce')

