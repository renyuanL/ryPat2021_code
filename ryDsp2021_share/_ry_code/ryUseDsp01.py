# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 13:33:55 2021

@author: renyu
"""

import os
import numpy as np

import matplotlib.pyplot as pl
import scipy.io


π= np.pi

#%%
import thinkdsp as td
#%%
from thinkdsp import *

#%%
# Generate a Wav Object
sample_rate= 10000 #samples/sec
duration= T= 1 # sec
ts= np.linspace(0,1,sample_rate)

freq= 110 #220 #440  # Hz
ys= np.cos(ts *2 *π *freq)


# write to file for playing using outside program "ffplay.exe"

data= ys
rate= sample_rate 
scipy.io.wavfile.write('_ry_sound.wav', rate, data)
os.system('ffplay _ry_sound.wav')

# plot it for visualization

pl.plot(ts, ys, color='red')

#%%

import subprocess

popen= subprocess.Popen(
    'ffplay _ry_sound.wav', 
    #shell= True
    )
popen.communicate()

#%%
cmd= 'ffplay _ry_sound.wav'
cmd= ['ffplay', '_ry_sound.wav']
subprocess.run(cmd)

#%%
#%%
#%%
#%%
#%%

#%%

rate, data= scipy.io.wavfile.read('rySound.wav')

scipy.io.wavfile.write('rySound_1.wav', rate*2, data)

rate1, data1= scipy.io.wavfile.read('rySound_1.wav')

#%%
w= Wave(ys, ts, framerate= sample_rate)

w.plot(color= 'red')
w.play()

#%%
# Read a wav file and make it an Wav Object

w= read_wave('rySound.wav')

w.plot(color= 'green')
w.play()
#%%
w.apodize()

w.plot(color= 'blue')
w.play()


#%%
# if we do not use thinkdsp.py, then ...

#%%
# Generate a Wav Object
sample_rate= 10000 # samples/sec

T= 1 #  (秒, sec)
ts= np.linspace(0,T, T * sample_rate)

freq= 110 #220 #440
ys= np.cos(ts *2 *π *freq)


pl.plot(ts,ys, color= 'red')



os.system('ffplay sound.wav')

#%%




