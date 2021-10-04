#!/usr/bin/env python
# coding: utf-8

# In[1]:
#
# Fitting to polynomials with NumPy
#

# Import required libraries NumPy, polynomial and matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Generate two random vectors
v1=np.random.rand(10)
v2=np.random.rand(10)

# Creates a sequence of equally separated values
sequence = np.linspace(v1.min(),v1.max(), num=len(v1)*10)

# Fit the data to polynomial fit data with 4 degree of polynomial
coefs = np.polyfit(v1, v2, 3)

# Evaluate polynomial on given sequence
polynomial_sequence = np.polyval(coefs,sequence)

# plot the  polynomial curve 
plt.plot(sequence, polynomial_sequence)

# Show plot
plt.show()

#%%

# Generate two random vectors
v1= np.random.rand(10)
v2= np.random.rand(10)
#%%ry

# Creates a sequence of equally separated values
sequence= np.linspace(
    v1.min(),
    v1.max(), 
    num= len(v1)*10)

# Fit the data to polynomial fit data with 4 degree of polynomial
degree= 8

coefs= np.polyfit(v1, v2, degree)

# Evaluate polynomial on given sequence
polynomial_sequence= np.polyval(coefs, sequence)

# plot the  polynomial curve 
plt.plot(sequence, polynomial_sequence)

plt.plot(v1,v2,'rx')
plt.title(f'degree= {degree}')
plt.grid()

# Show plot
plt.show()

#%%ry

for degree in range(0,10):

    coefs= np.polyfit(v1, v2, degree)
    
    # Evaluate polynomial on given sequence
    polynomial_sequence= np.polyval(coefs, sequence)
    
    # plot the  polynomial curve 
    plt.plot(sequence, polynomial_sequence)
    
    plt.plot(v1,v2,'rx')
    plt.title(f'degree= {degree}')
    plt.grid()
    
    # Show plot
    plt.show()


#%% ry

N= 1001

x= np.linspace(-10,10,N)

np.random.seed(0)
y0= np.random.normal(size=N)
y0 -= y0.mean()
y1= y0 + x
y2= y0 + x**2

plt.plot(x,y0,'rx')
plt.plot(x,y1,'go')
plt.plot(x,y2,'bs')

plt.title('ryTitle')
plt.grid()

#%% ry

# https://numpy.org/doc/stable/reference/routines.polynomials.html


'''
Polynomials

Polynomials in NumPy can be created, manipulated, and even fitted using the convenience classes of the numpy.polynomial package, introduced in NumPy 1.4.

Prior to NumPy 1.4, numpy.poly1d was the class of choice and it is still available in order to maintain backward compatibility. 
'''

# 新舊版本的 coef 順序反過來了！

degree= 2
coef0= np.polyfit(x, y0, deg= degree)

coef1= np.polynomial.Polynomial.fit(x, y0, deg= degree)


aPoly= np.polynomial.Polynomial(coef1)





# In[2]:

# https://numpy.org/doc/stable/reference/routines.linalg.html#module-numpy.linalg

# Import numpy 
import numpy as np

# Create matrix using NumPy
mat=np.mat([[2,4],[5,7]])
print("Matrix:\n",mat)

# Calculate determinant
print("Determinant:",np.linalg.det(mat))


# In[3]:



# Import numpy 
import numpy as np

# Create matrix using NumPy
mat= np.mat([[2,4],[5,7]])
print("Matrix:\n",mat)

# Find matrix inverse
inverse= np.linalg.inv(mat)
print("Inverse:\n",inverse)

#%% ry


# Create matrix using NumPy
A= [[2,4],
    [5,7]]

A= np.array(A)

# Find matrix det, inv, matrix rank, and matrix maultiplication


detA=   np.linalg.det(A)
invA=   np.linalg.inv(A)
A_invA= A@invA

rankA= np.linalg.matrix_rank(A)

print(f'''
A= 
{A}      

detA= 
{detA}

invA= 
{invA}

A_invA= 
{A_invA}

rankA= {rankA}
''')

assert np.allclose(
    A_invA,
    np.eye(A.shape[0]))

#%%
#
# Solving linear equations using NumPy
#
b= [1,-1]
b= np.array(b)

x= np.linalg.solve(A, b)
print(f'x= {x}')

assert np.allclose(
    A@x, 
    b)


#%%

# Decomposing a matrix using SVD

U,σ,V_T= np.linalg.svd(A)


print(f'''
U= 
{U},

σ= 
{σ},

V_T= 
{V_T}      
''')

assert np.allclose(
    A,
    U@np.diag(σ)@V_T
    )


#%%
#
# Eigenvectors and Eigenvalues using NumPy
#

λ, E= np.linalg.eig(A)

print(f'''
A=
{A}

λ= 
{λ},

E= 
{E},   
''')

assert np.allclose(
    A,
    E @ np.diag(λ) @ np.linalg.inv(E)
    )

#%%
B= A.T@A

λ1, E1= np.linalg.eig(B)

print(f'''
B=
{B},

λ1= 
{λ1},

E1= 
{E1},   
''')

assert np.allclose(
    B,
    E1 @ np.diag(λ1) @ np.linalg.inv(E1)
    )

#%%

L= np.linalg.cholesky(A@A.T)

#%% ry
#
# for symmtric matrix
#

C= np.array(
    [[2,1],
     [1,2]])

ans= {
    'rnk':  np.linalg.matrix_rank(C),
    'det':  np.linalg.det(C),
    'inv':  np.linalg.inv(C),
    'ch':   np.linalg.cholesky(C),
    'qr':   np.linalg.qr(C),
    'eig':  np.linalg.eig(C),
    'svd':  np.linalg.svd(C)
    }

print(ans)

#%%
C= np.array(
    [[3,2,1],
     [2,3,2],
     [1,2,3]])

ans= {
    'rnk':  np.linalg.matrix_rank(C),
    'det':  np.linalg.det(C),
    'inv':  np.linalg.inv(C),
    'ch':   np.linalg.cholesky(C),
    'qr':   np.linalg.qr(C),
    'eig':  np.linalg.eig(C),
    'svd':  np.linalg.svd(C)
    }

print(ans)

#%%

#%%

#%%

#%%



# In[4]:


# Create matrix A and Vector B using NumPy
A=np.mat([[1,1],[3,2]])
print("Matrix A:\n",A)

B = np.array([200,450]) 
print("Vector B:", B)


# In[5]:


# Solve linear equations
solution = np.linalg.solve(A, B) 
print("Solution vector x:", solution)


# In[6]:


# Check the solution
print("Result:",np.dot(A,solution))


# In[ ]:





# In[1]:


# Import numpy 
import numpy as np

# Create matrix using NumPy
mat=np.mat([[2,4],[5,7]])
print("Matrix:\n",mat)


# In[2]:


# Calculate the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(mat)
print("Eigenvalues:", eigenvalues) 
print("Eigenvectors:", eigenvectors) 


# In[3]:


# Compute eigenvalues 
eigenvalues= np.linalg.eigvals(mat)
print("Eigenvalues:", eigenvalues) 


# In[ ]:





# In[35]:


# import required libraries
import numpy as np
from scipy.linalg import svd

# Create a matrix
mat=np.array([[5, 3, 1],[5, 3, 0],[1, 0, 5]])

# Perform matrix decomposition using SVD 
U, Sigma, V_transpose = svd(mat)

print("Left Singular Matrix:",U)
print("Diagonal Matrix: ", Sigma)
print("Right Singular Matrix:", V_transpose)


# In[42]:


# import required libraries
import numpy as np
from numpy.linalg import matrix_rank

# Create a matrix
mat=np.array([[5, 3, 1],[5, 3, 1],[1, 0, 5]])

# Compute rank of matrix
print("Matrix: \n", mat)
print("Rank:",matrix_rank(mat))


# In[ ]:





# In[ ]:





# In[11]:


# Import numpy 
import numpy as np

# Create an array with random values
random_mat=np.random.random((3,3))
print("Random Matrix: \n",random_mat)


# In[ ]:





# In[4]:


# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

# Create an numpy vector of size 5000 with value 0
cash_balance = np.zeros(5000) 

cash_balance[0] = 500

# Generate random numbers using Binomial
samples = np.random.binomial(9, 0.5, size=len(cash_balance))

# Update the cash balance
for i in range(1, len(cash_balance)):
    if samples[i] < 5:
        cash_balance[i] = cash_balance[i - 1] - 1 
    else:
        cash_balance[i] = cash_balance[i - 1] + 1 

# Plot the updated cash balance
plt.plot(np.arange(len(cash_balance)), cash_balance)
plt.show()


# In[7]:


# Import required library
import numpy as np
import matplotlib.pyplot as plt 

sample_size=225000

# Generate random values sample using normal distriubtion
sample = np.random.normal(size=sample_size)

# Create Histogram 
n, bins, patch_list = plt.hist(sample, int(np.sqrt(sample_size)), density=True) 

# Set parameters
mu, sigma=0,1

x= bins
y= 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) )

# Plot line plot(or bell curve)
plt.plot(x,y,color='red',lw=2)
plt.show()


# In[9]:


# create small, medium, and large samples for noymality test
small_sample = np.random.normal(loc=10, scale=6, size=10)
medium_sample = np.random.normal(loc=10, scale=6, size=100)
large_sample = np.random.normal(loc=10, scale=6, size=1000)


# In[10]:


# Histogram for small
import seaborn as sns
import matplotlib.pyplot as plt

# Create distribution plot
sns.distplot(small_sample)

plt.show()


# In[11]:


# Histogram for medium
import seaborn as sns
import matplotlib.pyplot as plt

# Create distribution plot
sns.distplot(medium_sample)

plt.show()


# In[12]:


# Histogram for large
import seaborn as sns
import matplotlib.pyplot as plt

# Create distribution plot
sns.distplot(large_sample)

plt.show()


# In[13]:


# Import shapiro funtion
from scipy.stats import shapiro

# Apply Shapiro-Wilk Test
print("Shapiro-Wilk Test for Small Sample: ",shapiro(small_sample))
print("Shapiro-Wilk Test for Medium Sample: ",shapiro(medium_sample))
print("Shapiro-Wilk Test for Large Sample: ",shapiro(large_sample))


# In[14]:


# Import anderson funtion
from scipy.stats import anderson

# Apply Anderson-Darling Test
print("Anderson-Darling Test for Small Sample: ",anderson(small_sample))
print("Anderson-Darling Test for Medium Sample: ",anderson(medium_sample))
print("Anderson-Darling Test for Large Sample: ",anderson(large_sample))


# In[15]:


# Import normaltest function
from scipy.stats import normaltest

# Apply  D'Agostino-Pearson test
print("D'Agostino-Pearson Test for Small Sample: ", normaltest(small_sample))
print("D'Agostino-Pearson Test  for Medium Sample: ",normaltest(medium_sample))
print("D'Agostino-Pearson Test  for Large Sample: ",normaltest(large_sample))


# In[16]:


# Import required library
import numpy as np
from scipy.misc import face
import matplotlib.pyplot as plt

face_image = face()
mask_random_array = np.random.randint(0, 3, size=face_image.shape)

fig, ax = plt.subplots(nrows=2, ncols=2)

# Display the Original Image 
plt.subplot(2,2,1)
plt.imshow(face_image)
plt.title("Original Image")
plt.axis('off')

# Display masked array
masked_array = np.ma.array(face_image, mask=mask_random_array)
plt.subplot(2,2,2)
plt.title("Masked Array")
plt.imshow(masked_array)
plt.axis('off')

# Log operation on original image
plt.subplot(2,2,3)
plt.title("Log Operation on Original")
plt.imshow(np.ma.log(face_image).astype('uint8'))
plt.axis('off')

# Log operation on masked array
plt.subplot(2,2,4)
plt.title("Log Operation on Masked")
plt.imshow(np.ma.log(masked_array).astype('uint8'))
plt.axis('off')

# Display the subplots
plt.show()

