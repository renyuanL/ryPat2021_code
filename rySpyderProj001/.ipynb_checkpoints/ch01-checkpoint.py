# -*- coding: utf-8 -*-
"""
Ch01: Linear Algebra

Created on Sat Sep 25 17:36:35 2021

@author: renyu


"""

x= 1
y= 1.0
z= 1+1j
#%%
aList= [1,2,3]
aSet=  {10,20,30}
aDict= {'a':1, 'b':2, 'c':3}
#%%
import numpy as np

v= [10, 20, 30]
v= np.array(v)


A= [[1, 2, 3],
    [4, 5, 6]]

A= np.array(A)
#%%
v1= [10, 20, 30]
v2= [40, 50, 60]

v1= np.array(v1)
v2= np.array(v2)
#%%
B= np.arange(2*3*4)
#%%
B= B.reshape(2,3,4)

#%%
import sympy as sy
x,y,z= sy.symbols('x,y,z')

from sympy import MatrixSymbol, Matrix

#%%
# Solve linear equations
########################

A= [[1,-2],
    [2, 1]]
A= np.array(A)

b= [1,7]
b= np.array(b)
b= b.reshape(2,1)

#[sol]
x= np.linalg.solve(A,b)

# 驗算
A@x == b 
np.isclose(A@x, b) 
#%%

# [sol2[]

x1= np.linalg.inv(A) @ b

# 驗算
A@x1 == b
np.isclose(A@x1, b) 

#%%
# n 變數 線性方程組

n= 3 #np.random.randint(2,10)

np.random.seed(0)
A= np.random.randint(-5,5,size=(n,n))

np.random.seed(1)
b= np.random.randint(-5,5,size=(n,1))

x= np.linalg.solve(A, b)

print(
f'''
n= 
{n}, 

A= 
{A}

b= 
{b}

x= 
{x}
''')
#%%

invA= np.linalg.inv(A)

x1= invA @ b

print(
f'''
n= 
{n}, 

A= 
{A}

invA=
{invA}

b= 
{b}

x1= 
{x1}
''')

#%%
# 驗算
validation= np.isclose(A@x, b)
print(validation)

validation= np.isclose(A@x1, b)
print(validation)
#%%
# HomeWork:
# 自己寫一個遞迴式的 function，來計算 Determinant 


n= 4 

np.random.seed(0)
A= np.random.randint(-5,5,size=(n,n))
print(f'A= \n{A}')

det= np.linalg.det(A)
print(f'det= {det}')

#%%
# Determinant

i= 0
det= 0
for j in range(n):
    Bi=  np.delete(A, i, axis=0)
    Bij= np.delete(Bi,j, axis=1)
    Bij= np.linalg.det(Bij)
    Aij= A[i,j]*Bij*(-1)**(i+j)
    det += Aij
print(f'det= {det}')

#%%

j= 0
det= 0
for i in range(n):
    Bj=  np.delete(A, j, axis=1)
    Bij= np.delete(Bj,i, axis=0)
    Bij= np.linalg.det(Bij)
    Aij= A[i,j]*Bij*(-1)**(i+j)
    det += Aij
print(f'det= {det}')

#%%

# when n is large, it is very slow!!

def ryDet(A):
    
    n= A.shape[0]
    if n>1:
        i= 0
        det= 0
        for j in range(n):
            Bi=  np.delete(A, i, axis=0)
            Bij= np.delete(Bi,j, axis=1)
            
            Bij= ryDet(Bij) #np.linalg.det(Bij)
            
            Aij= A[i,j]*Bij*(-1)**(i+j)
            det += Aij
        #print(f'det= {det}')
        return det
    
    elif A.shape[0]==1:
        det= A[0,0]
        return det
    else:
        return None

det= ryDet(A)
print(f'det= {det}')
#%%

#%%

print(f'A= \n{A}')

i= 0
Bi=  np.delete(A, i, axis=0)
print(f'i= {i}, Bi= \n{Bi}')

det= 0
for j in range(n):    
    Bij= np.delete(Bi,j, axis=1)
    print(f'i,j= {i}{j}, Bij= \n{Bij}')
    Dij= np.linalg.det(Bij)
    print(f'.... Dij= {Dij}')
    Cij= A[i,j]*Dij*(-1)**(i+j)
    det += Cij

print(f'det= {det}')

#%%
# when n is large, it is very slow!!

print(f'A= \n{A}')

def ryDet(A):
    
    n= A.shape[0]
    if n>1:
        i= 0
        Bi=  np.delete(A, i, axis=0)
        #print(f'\ni= {i}, Bi= \n{Bi}')
        
        det= 0
        for j in range(n):
            
            Bij= np.delete(Bi,j, axis=1)
            print(f'\ni,j= {i}{j}, Bij= \n{Bij}')
            
            Dij= ryDet(Bij) #np.linalg.det(Bij)
            print(f'.... Dij= {Dij}')
            
            Cij= A[i,j]*Dij*(-1)**(i+j)
            det += Cij
        #print(f'det= {det}')
        return det
    
    elif A.shape[0]==1:
        det= A[0,0]
        return det
    else:
        return None

det= ryDet(A)
print(f'\ndet= {det}')
#%%

import scipy.linalg

A= [[4,3],
    [6,3]]

'''
A= [[12, -51, 4], 
    [6, 167, -68],
    [-4, 24, -41]]
'''

A= np.array(A)

P,L,U= scipy.linalg.lu(A)

# 驗算
P @ L @ U == A
np.isclose(P @ L @ U , A)
np.allclose(P @ L @ U , A)

#%%
Q,R= scipy.linalg.qr(A)

# 驗算
(Q @ R == A).all()

np.allclose(Q @ R , A)

# 再檢查 Q 的正交性質
#assert Q[0]@Q[1] == 0
#assert Q[0]@Q[0] == 1
assert np.allclose(Q[0]@Q[1] , 0)
assert np.allclose(Q[0]@Q[0] , 1)

#%%
λ, Q= scipy.linalg.eig(A)

lmbd= λ 
# 驗算
#assert A @ Q[:,0] == Q[:,0] * λ[0]

assert np.allclose(A @ Q[:,0], Q[:,0] * λ[0] )

assert np.allclose(A @ Q, Q @ np.diag(λ) )
#%%
U, σ, V_T= scipy.linalg.svd(A)

V= V_T.T

Σ= np.diag(σ)

sigma= σ
Sigma= Σ 

# 驗算

assert np.allclose(U @  Σ @ V.T, A )

# 再檢查 U, V 的正交性質

assert np.allclose(U[0]@U[1] , 0)
assert np.allclose(V[0]@V[1] , 0)
assert np.allclose(U[0]@U[0] , 1)
assert np.allclose(V[0]@V[0] , 1)

#
# 檢查 orthogonal , orthonormal
#
assert np.allclose(U@U.T, np.eye(U.shape[0]))
assert np.allclose(V@V.T, np.eye(V.shape[0]))

#
assert np.allclose(A  @V, U@Σ)
assert np.allclose(A.T@U, V@Σ)

#%%
import numpy as np
import scipy.linalg as sp_linalg

M= [[1,2],
    [3,4]]
M= np.array(M)
U,sigma,V_T= np.linalg.svd(M)
V= V_T.T
MtM= M.T@M
ld, Q= np.linalg.eig(MtM)
#%%

A= [[12, -51, 4], 
    [6, 167, -68],
    [-4, 24, -41]]
A= np.array(A)
Q,R= np.linalg.qr(A)

#%%
A= [[2,1],
    [1,2]]
A= np.array(A)
lmda, eigV= np.linalg.eig(A)
#%%

Q, R= np.linalg.qr(A)

#%%
U,Sigma,V_T= np.linalg.svd(A)
V= V_T.T

#%%
A= [[1,-1],
    [1,1]]
A= np.array(A)
S= np.diag([2,1])
B= A@S@A.T

print(f'A= \n{A}, \nS= \n{S}, \nB= \n{B}')
#%%
lmda, eigV= np.linalg.eig(B)
Q, R= np.linalg.qr(B)
U,Sigma,V_T= np.linalg.svd(B)
V= V_T.T

#%%

A= [[10,1],
    [1,10]]
A= np.array(A)

lmda, eigV= np.linalg.eig(A)

#%%

A= [[1,1],
    [0,1]]
A= np.array(A)

lmda, eigV= np.linalg.eig(A)




#%%

A= [[4,1],
    [1,4]]
A= np.array(A)

X= [[1,0,1,-1,-1,-1],
    [0,1,1,+1,-1,+1]]
X= np.array(X)

Y= A@X

