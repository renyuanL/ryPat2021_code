# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 21:02:59 2021

@author: renyu
"""


filename= 'rySound.wav'

#%%
with open(filename, mode= 'rb') as fp:
    hdr= fp.read(44)
    x1=  fp.read()

#%%
import wave

with wave.open(filename, "rb") as fp:

    nchannels= fp.getnchannels()
    nframes=   fp.getnframes()
    sampwidth= fp.getsampwidth()
    framerate= fp.getframerate()
    
    x2= fp.readframes(nframes)


#%%
import numpy as np
import matplotlib.pyplot as pl
    
ys= np.frombuffer(x2, dtype= np.int16)

pl.plot(ys)

#%%
import scipy.io.wavfile  as wf

sample_rate, ys= wf.read(filename)

pl.plot(ys)

#%%
import librosa as lb

ys1, sample_rate= lb.load(filename, sr= None)

pl.plot(ys1)

#%%

# struct — Interpret bytes as packed binary data

# https://docs.python.org/3/library/struct.html

import struct

hdr00_12=   struct.unpack('12s', hdr[ 0:12])
hdr12_36=   struct.unpack('24s', hdr[12:36])
hdr36_44=   struct.unpack('8s',  hdr[36:44])
dat00_10=   struct.unpack('10h',  x1[ 0:20])

'''
hdr12_36
Out[103]: (b'fmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00',)

hdr00_12
Out[104]: (b'RIFFtu\r\x00WAVE',)

hdr12_36
Out[105]: (b'fmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00',)

hdr36_44
Out[106]: (b'dataPu\r\x00',)

dat00_10
Out[107]: (-3, 18, -1, -15, 4, 6, -12, 18, 9, 2)
'''
#
# compare data0 with ys[0:10]
#
'''
dat00_10
Out[78]: (-3, 18, -1, -15, 4, 6, -12, 18, 9, 2)

ys[0:10]
Out[79]: array([ -3,  18,  -1, -15,   4,   6, -12,  18,   9,   2], dtype=int16)
'''

