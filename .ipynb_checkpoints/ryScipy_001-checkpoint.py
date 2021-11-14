# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 08:43:30 2021

@author: renyu
"""
import scipy.stats as st

import inspect

#%%

st.randint
st.binom
st.poisson
st.bernoulli
st.geom
st.nbinom


st.uniform
st.norm


#%%
aL= [a 
     for a in dir(st.rv_discrete)
     if not a.startswith('_')
     ]

#%%
# 列出所有 離散型 隨機變數
dist_discrete= [
    x 
    for x in dir(st) 
    if isinstance(
        getattr(st, x), 
        st.rv_discrete)
]

#%%
dL= []
cL= []

for x in dir(st):
    y= getattr(st, x)
    
    if isinstance(y, st.rv_discrete):
        
        s= inspect.getsource(y._pmf)
        s+= inspect.getsource(y._logpmf)
        s+= inspect.getsource(y._cdf)
        
        dL += [
            (x,y, s)
            ]
    if isinstance(y, st.rv_continuous):
        
        s= inspect.getsource(y._pdf)
        s += inspect.getsource(y._logpdf)
        s += inspect.getsource(y._cdf)
        
        cL += [
            (x,y, s)
            ]
#%%

#%%
from ryPat import *    

#ryHistogram_demo()
#ryDistribution_demo('normal')
#%%
aL= [
    x
    for x in dir(np.random) 
    if not x.startswith('_')
    ]

#%%
import numpy as np

rg= np.random.default_rng()
X= rg.binomial(n=10, p=.5, size=100)
f,x= np.histogram(X, bins=11)

#%%
import sys
sys.path.insert(0,'L:\\OneDrive - 長庚大學\\_ry_2021_fall_CguOneDrive\\_ryDsp2021\\ryDsp2021')
#%%
import ryDsp
sr, wav= ryDsp.getWavFromYoutube()

sr= sr//100
wav= wav[::100, 0]
#%%
ryDsp.plotWav(wav)
#%%
import yfinance as yf
googl= yf.Ticker('googl')
googl= googl.history(period='max')
googl= googl['Close']
googl= googl.to_numpy()
#%%
