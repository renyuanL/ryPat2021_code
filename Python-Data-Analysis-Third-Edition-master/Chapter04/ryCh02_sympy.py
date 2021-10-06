# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 10:25:35 2021

@author: renyu
"""
import sympy as sy
#from sympy import *
#%%
x, y, z, a, b, c= sy.symbols('x, y, z, a, b, c')
#%%
f= a*x**2 + b*x +c

#%%
df= sy.diff(f, x)
print(df)
#%%
df_x= f.diff(x)
F=    f.integrate(x)

print(f'''
f= 
{f}

df_x= 
{df_x}

F= 
{F}
''')
      
#%%

#%%

f= f*f

df_x= f.diff(x)
F=    f.integrate(x)

print(f'''
f= 
{f}

df_x= 
{df_x}

F= 
{F}
''')

#%%
#%%
#%%
#%%
