#!/usr/bin/env/python3
# -*- coding: utf-8 -*-
import os
from scipy import signal
import numpy as np
import pandas as pd
import sys
from plotly import io
import plotly.express as px
from natsort import natsorted

colorscale = [
    [0, 'rgb(133,0,0)'], #2
    [0.076, 'rgb(202,0,0)'], #3
    [0.153, 'rgb(255,16,0)'], #4
    [0.230, 'rgb(255,96,0)'], #5
    [0.307, 'rgb(255,175,0)'], #6
    [0.384, 'rgb(238,255,17)'], #7
    [0.461, 'rgb(175,255,80)'], #8
    [0.538, 'rgb(91,255,163)'], #9
    [0.615, 'rgb(16,255,239)'], #10
    [0.692, 'rgb(0,175,255)'], #11
    [0.769, 'rgb(0,111,255)'], #12
    [0.846, 'rgb(0,32,255)'], #13
    [0.923, 'rgb(0,0,216)'], #14
    [1, 'rgb(0,0,158)'] #15
]

# using the ASTM C1383-15(2022) Standard Test Method for Measuring the P-Wave Speed and the Thickness of Concrete Plates Using the Impact-Echo Method

def peakFrequencyMapping(filelist, f_s, flag):
    
    for entry in filelist:
        f = open(path_+'/'+entry, "r")
        data = f.read()
        data = data.split('\n')
        if data[-1] == "":
            data = data[:-1]
        data_arr = []
        for i in range(0,len(data)):
                temp = data[i].split('\t')
                data_arr.append(float(temp[1]))
        entry = entry.strip('.txt')
        entry = entry.split('-')
        YLocation.append(entry[0]) #get x location from filename
        XLocation.append(entry[1]) #get y location from filename
        data_detrended = signal.detrend(data_arr) #linear detrending
        Amplitude = np.fft.fft(data_detrended)
        freq = np.fft.fftfreq(data_detrended.shape[-1]) * f_s
        for i in range(0,f_s):
            if abs(Amplitude[i]) == max(abs(Amplitude[0:f_s])):
                FrequencyVal.append(freq[i]) #peak freqeuncy value
    
    index = natsorted(list(set(YLocation)))
    cols = natsorted(list(set(XLocation)))
    fdf = pd.DataFrame(np.reshape(FrequencyVal, (len(index),len(cols))), index = index, columns = cols)
    if flag == 0:
        fig = px.imshow(fdf, text_auto = False, color_continuous_scale = colorscale, zmin=2, zmax=15)
        fig.update_layout(coloraxis_colorbar=dict(len=0.5, thickness=15))
        io.write_image(fig, 'output.png')
    elif flag == 1:
        fig = px.imshow(fdf, text_auto = True, color_continuous_scale = colorscale, zmin=2, zmax=15)
        fig.update_layout(coloraxis_colorbar=dict(len=0.5, thickness=15))
        io.write_image(fig, 'output.png')
    else:
        print('flag error')
    
    
    
    
if __name__ == "__main__":
    
    for a in sys.argv:
        if a == '-i':
            path_ = sys.argv[2]
        if a == '-f':
            f_s_ = int(sys.argv[4])
        if a == '-a':
            annotation_ = int(sys.argv[6])
    
    filelist_ = []
    files  = os.listdir(path_)
    ext = [".dat", ".txt"]
    for file in files:
        if file.endswith(tuple(ext)):
            filelist_.append(file)  
    YLocation = []
    XLocation = []
    FrequencyVal = []
        
    peakFrequencyMapping(filelist_, f_s_, annotation_)


