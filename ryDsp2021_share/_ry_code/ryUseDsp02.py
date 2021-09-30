# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 17:51:54 2021

@author: renyu
"""

import os
import subprocess

import numpy as np

import matplotlib.pyplot as pl
import scipy.io.wavfile  as wf


π= np.pi

#%%
# Generate a Wav Object
sample_rate= 10000 #samples/sec
duration= T= 1 # sec
ts= np.linspace(0,1,sample_rate)

freq= 110 #220 #440  # Hz
ys=   np.cos(ts *2 *π *freq)


# write to file for playing using outside program "ffplay.exe"

#data= ys
#rate= sample_rate 
wf.write('_tmp_.wav', sample_rate, ys)


# use outside program to play the wav

cmd= 'ffplay _tmp_.wav'
subprocess.run(cmd) # "ffplay.exe" should exist in your system

#%%
# play an exist wav file

# use outside program to play the wav

cmd= 'ffplay rySound.wav'
subprocess.run(cmd) # "ffplay.exe" should exist in your system

#%%
sample_rate, ys= wf.read('rySound.wav')

#%%
pl.plot(ys)
#%%
dataDuration= len(ys)/sample_rate  # 秒, sec
#%%

#from thinkdsp import apodize

#
# rewrite the "apodize" function in thindsp.py
# 
def apodize(ys, framerate, denom=20, duration=0.1):
    """Tapers the amplitude at the beginning and end of the signal.

    Tapers either the given duration of time or the given
    fraction of the total duration, whichever is less.

    ys: wave array
    framerate: int frames per second
    denom: float fraction of the segment to taper
    duration: float duration of the taper in seconds

    returns: wave array
    """
    
    # a fixed fraction of the segment
    n= len(ys)
    k1= n // denom

    # a fixed duration of time
    k2= int(duration * framerate)

    k=  min(k1, k2)

    w1= np.linspace(0, 1, k)
    w2= np.ones(n - 2 * k)
    w3= np.linspace(1, 0, k)

    window= np.concatenate((w1, w2, w3))
    
    # ry modified
    ys= ys*window
    ys= ys/abs(ys).max()
    
    return ys

ys= apodize(ys, sample_rate, denom= 10, duration= 1)

pl.plot(ys)

#%%

wf.write('_tmp_.wav', sample_rate, ys)

# use outside program to play the wav

cmd= 'ffplay _tmp_.wav'
subprocess.run(cmd) # "ffplay.exe" should exist in your system

#%%


from thinkdsp_def import *






