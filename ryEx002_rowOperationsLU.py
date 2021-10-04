# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 17:03:00 2021

@author: renyu
"""

import numpy as np
import scipy.linalg as sp_linalg
import pandas as pd


#%% ry: Row Operation for 3x3 matrix

A0= np.array(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 10]])

A= A0.copy()

assert A[0,0] !=0 # otherwise, rowChange(A,0)

c10= -A[1,0]/A[0,0]
P10= np.array(
    [[1, 0, 0],
     [c10, 1, 0],
     [0, 0, 1]])

A= P10@A


c20= -A[2,0]/A[0,0]
P20= np.array(
    [[1, 0, 0],
     [0, 1, 0],
     [c20, 0, 1]])

A= P20@A

assert A[1,1] !=0  # otherwise, rowChange(A,1)

c21= -A[2,1]/A[1,1]
P21= np.array(
    [[1, 0, 0],
     [0, 1, 0],
     [0, c21, 1]])
A= P21@A

A1= A

P= P21@P20@P10
invP= np.linalg.inv(P)

print(f'''
A0= 
{A0}

invP= 
{invP}

A1= 
{A1}   
''')

#%%

#%%


A1= np.array(
    [[1]])

A2= np.array(
    [[2,1],
     [1,2]])

A3= np.array(
    [[3, 2, 1],
     [2, 3, 2],
     [1, 2, 3]])

#%%

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
