# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 14:13:13 2021

@author: renyu
"""
import ryPat

ryPat.ryGradient_demo()

#%%

from ryPat import *

def f(x,y):
    z= sm.exp(-(x**2+y**2))
    return z

z= f(x,y)

ryPat.ryGradient_demo(z)

#%%

def f1(x,y):
    z= sm.sin(x)/x + sm.sin(y)/y
    return z

z1= f1(x,y)

ryPat.ryGradient_demo(z1)

#%%

z2=   f(x,y) 
z2 += f(x-5, y-5)*2 
z2 += f((x-2)/2, (y+5)/5) /4

ryPat.ryGradient_demo(z2)
#%%
