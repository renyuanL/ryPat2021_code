# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:52:06 2021

@author: renyu
"""

import numpy as np

import thinkdsp_def   as td

import thinkdsp_class as tc

from thinkdsp_def   import *
from thinkdsp_class import *

#%%
x= 8
xs= [1,2,3,4,5,6,7,8,9]

i= td.find_index(x,xs)
print(i)
#%%

ys= np.arange(0,100)

print(
  shift_right(ys,10)
)

print(
  shift_left(ys,10)
)
#%%

t= np.linspace(0,10,1001)
x= np.cos(2*np.pi*100*t)

x= Wave(x,t)
X= x.make_spectrum()

#%%
#%%
#%%
#%%
#%%
#%%
#%%



