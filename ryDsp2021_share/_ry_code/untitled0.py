# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 13:33:55 2021

@author: renyu
"""
import numpy as np

import thinkdsp as td
#%%
from thinkdsp import *
#%%
ts= np.linspace(0,1,1001)
ys= np.cos(ts*2*np.pi*1)

