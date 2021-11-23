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

#ryPat.ryGradient_demo(z)

#%%

def f1(x,y):
    z= sm.sin(x)/x + sm.sin(y)/y
    return z

z1= f1(x,y)

#ryPat.ryGradient_demo(z1)

#%%

z2=   f(x,y) 
z2 += f(x-5, y-5)*2 
z2 += f((x-2)/2, (y+5)/5) /4

#ryPat.ryGradient_demo(z2)
#%%
# This function can be a cdf and its derivative can be a pdf
# but the mean (expectation value) will go to infty

def f2(x):
    y= sm.atan(x)/sm.pi +1/2
    return y

z= f2(x)
#ryPat.ryPlotGradientMap(z)

def ϕ(x):
    Y= f2(x) #sm.atan(x)/sm.pi +1/2
    y= Y.diff(x)
    return y

z= ϕ(x)
#ryPat.ryPlotGradientMap(z)
    
#%%
# 試試看這條 神奇的 lambda 語法
a,b,c= sm.symbols('a,b,c')

q= (lambda *x: 
    sum([s*t 
     for s in x 
     for t in x]))
    
print( 
f'''q= {q}, 
q(a,b,c)= {q(a,b,c)}''' )

a= sm.symbols('a0:10')
q(*a)

print( 
f'''a= {a}, 
q(*a)= {q(*a)}''' )


#%%